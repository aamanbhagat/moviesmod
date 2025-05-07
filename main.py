import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import sys
import os

import aiofiles
from playwright.async_api import async_playwright, Browser, Page, TimeoutError
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("url_processor")

# Constants
INPUT_FILE = "/workspaces/codespaces-blank/merged.txt"
OUTPUT_FILE = "/workspaces/codespaces-blank/output.txt"
FINISHED_FILE = "/workspaces/codespaces-blank/wow.txt"
MAX_WORKERS = 10  # Adjust based on your machine capabilities (4 cores)
WAIT_TIMEOUT = 30000  # 30 seconds in ms
RETRY_INTERVAL = 10  # 10ms
MAX_RETRIES = 2  # Maximum number of retries for a failed URL

# Process status tracking
class ProcessStatus:
    def __init__(self):
        self.workers: Dict[int, Dict[str, Any]] = {}
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.retried = 0
        self.total = 0
        self.console = Console()
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
        )
        self.overall_task: Optional[TaskID] = None
        self.finished_urls: Set[str] = set()
        
    def initialize(self, total: int):
        self.total = total
        self.overall_task = self.progress.add_task("[yellow]Overall Progress", total=total)
        
    def update_worker(self, worker_id: int, url: str, status: str, current_step: str):
        self.workers[worker_id] = {
            "url": url,
            "status": status,
            "step": current_step,
            "last_update": time.time()
        }
    
    def increment_processed(self, success: bool = True):
        self.processed += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1
        if self.overall_task is not None:
            self.progress.update(self.overall_task, completed=self.processed)
    
    def add_finished_url(self, url: str):
        self.finished_urls.add(url)
    
    def increment_retried(self):
        self.retried += 1
    
    def generate_table(self) -> Table:
        table = Table(title="URL Processing Status")
        table.add_column("Worker ID", justify="center", style="cyan")
        table.add_column("URL", style="green", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Current Step", justify="center")
        
        for worker_id, info in self.workers.items():
            url_display = info["url"]
            if len(url_display) > 50:
                url_display = url_display[:47] + "..."
            
            status_style = "green" if info["status"] == "Processing" else "red"
            table.add_row(
                str(worker_id),
                url_display,
                f"[{status_style}]{info['status']}[/{status_style}]",
                info["step"]
            )
            
        # Add summary row
        table.add_section()
        table.add_row(
            "",
            f"Processed: {self.processed}/{self.total}",
            f"Success: {self.successful}",
            f"Failed: {self.failed}",
            f"Retried: {self.retried}"
        )
        
        return table

# Main status tracker
status_tracker = ProcessStatus()

async def wait_and_click(page: Page, selector: str, worker_id: int, step_name: str) -> bool:
    """Wait for selector to be available and click it with retry logic"""
    status_tracker.update_worker(worker_id, page.url, "Processing", f"Waiting for: {step_name}")
    
    start_time = time.time()
    while time.time() - start_time < WAIT_TIMEOUT/1000:
        try:
            element = await page.query_selector(selector)
            if element:
                await element.click()
                await page.wait_for_timeout(500)  # Wait a bit after clicking
                return True
        except Exception as e:
            pass
            
        await asyncio.sleep(RETRY_INTERVAL/1000)
    
    return False

async def extract_download_url(page: Page, worker_id: int) -> Optional[str]:
    """Extract download URL with retry logic - every 10ms for up to 30 seconds"""
    status_tracker.update_worker(worker_id, page.url, "Processing", "Extracting download URL")
    
    start_time = time.time()
    attempt_count = 0
    
    while time.time() - start_time < WAIT_TIMEOUT/1000:
        attempt_count += 1
        
        if attempt_count % 50 == 0:  # Log every ~500ms
            elapsed = time.time() - start_time
            status_tracker.update_worker(worker_id, page.url, "Processing", 
                                       f"Extracting URL (Attempt {attempt_count}, {elapsed:.1f}s)")
            logger.info(f"Worker {worker_id}: Still trying to extract URL - {elapsed:.1f}s elapsed")
        
        # Try different extraction methods in sequence
        
        # Method 1: Direct selector
        try:
            element = await page.query_selector('a#two_steps_btn')
            if element:
                href = await element.get_attribute('href')
                if href:
                    logger.info(f"Worker {worker_id}: Extracted URL via direct selector after {attempt_count} attempts")
                    return href
        except Exception:
            pass

        # Method 2: JavaScript evaluation
        try:
            href = await page.evaluate("""
                () => {
                    const element = document.querySelector('a#two_steps_btn');
                    return element ? element.href : null;
                }
            """)
            if href:
                logger.info(f"Worker {worker_id}: Extracted URL via JavaScript after {attempt_count} attempts")
                return href
        except Exception:
            pass
            
        # Method 3: XPath
        try:
            element = await page.query_selector('//a[@id="two_steps_btn"]')
            if element:
                href = await element.get_attribute('href')
                if href:
                    logger.info(f"Worker {worker_id}: Extracted URL via XPath after {attempt_count} attempts")
                    return href
        except Exception:
            pass
            
        # Method 4: Find by content
        if attempt_count % 30 == 0:  # Check less frequently as it's more expensive
            try:
                links = await page.evaluate("""
                    () => {
                        return Array.from(document.querySelectorAll('a'))
                            .filter(a => a.href && a.innerText && 
                                    (a.innerText.includes('Go to download') || 
                                     a.innerText.includes('Download')))
                            .map(a => a.href);
                    }
                """)
                if links and len(links) > 0:
                    logger.info(f"Worker {worker_id}: Extracted URL via text search after {attempt_count} attempts")
                    return links[0]
            except Exception:
                pass
                
        # Method 5: Every 100 attempts, try scrolling to reveal potentially hidden elements
        if attempt_count % 100 == 0:
            try:
                await page.evaluate("window.scrollBy(0, 100)")
            except Exception:
                pass
                
        # Short wait before next attempt
        await asyncio.sleep(RETRY_INTERVAL/1000)
    
    logger.warning(f"Worker {worker_id}: Failed to extract URL after {attempt_count} attempts")
    return None

async def process_url(browser: Browser, url: str, worker_id: int, output_queue: asyncio.Queue, finished_queue: asyncio.Queue, retry_queue: asyncio.Queue, retry_count: int = 0):
    """Process a single URL with retry capability"""
    page = None
    final_url = None
    context = None
    
    try:
        # Create a new context for each page (for isolation)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        )
        
        # Block ads and trackers
        await context.route("**/*", lambda route: route.abort() 
                           if any(ad_domain in route.request.url for ad_domain in [
                               'googleadservices.com', 'googlesyndication.com', 
                               'doubleclick.net', 'adnxs.com', 'facebook.com'
                           ]) else route.continue_())
        
        page = await context.new_page()
        
        # Update status
        status_tracker.update_worker(worker_id, url, "Processing", "Opening URL")
        
        # Step 1: Navigate to the URL
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_timeout(2000)  # Wait for JS to execute
        
        # Step 2: Click on "Start Verification"
        status_tracker.update_worker(worker_id, url, "Processing", "Clicking Start Verification")
        start_verification_clicked = await wait_and_click(
            page, "h5:has-text('Start Verification')", worker_id, "Clicking Start Verification"
        )
        if not start_verification_clicked:
            # Try alternate methods
            try:
                elements = await page.query_selector_all("h5")
                for element in elements:
                    text = await element.text_content()
                    if "Start Verification" in text:
                        await element.click()
                        start_verification_clicked = True
                        break
            except Exception:
                pass
                
            if not start_verification_clicked:
                logger.warning(f"Worker {worker_id}: Could not find 'Start Verification' button")
        
        # Step 3: Click on verify_button2
        status_tracker.update_worker(worker_id, url, "Processing", "Clicking verify_button2")
        await page.wait_for_timeout(2000)  # Wait before next click
        verify_button2_clicked = await wait_and_click(
            page, "span#verify_button2", worker_id, "Clicking verify_button2"
        )
        if not verify_button2_clicked:
            logger.warning(f"Worker {worker_id}: Could not find 'verify_button2'")
        
        # Step 4: Click on verify_button
        status_tracker.update_worker(worker_id, url, "Processing", "Clicking verify_button")
        await page.wait_for_timeout(2000)  # Wait before next click
        verify_button_clicked = await wait_and_click(
            page, "span#verify_button", worker_id, "Clicking verify_button"
        )
        if not verify_button_clicked:
            logger.warning(f"Worker {worker_id}: Could not find 'verify_button'")
        
        # Make sure page is fully loaded after all button clicks
        await page.wait_for_load_state("networkidle", timeout=10000)
        await page.wait_for_timeout(3000)  # Additional wait to ensure content is loaded
        
        # Step 5: Extract URL from download button - using the specific selector with retry logic
        download_url = await extract_download_url(page, worker_id)
        
        if not download_url:
            logger.error(f"Worker {worker_id}: Could not extract download URL after all attempts")
            raise Exception("Download URL extraction failed after 30 seconds of attempts")
            
        # Step 6: Navigate to that URL and extract final URL
        status_tracker.update_worker(worker_id, url, "Processing", "Navigating to download URL")
        response = await page.goto(download_url, wait_until="domcontentloaded")
        if response:
            logger.info(f"Worker {worker_id}: Navigation status: {response.status}")
            
        await page.wait_for_load_state("networkidle", timeout=10000)
        
        # Extract the final URL
        final_url = page.url
        logger.info(f"Worker {worker_id}: Final URL: {final_url}")
        
        # Put result in output queue - only the final URL
        await output_queue.put(final_url)
        
        # Put original URL in the finished queue for wow.txt
        await finished_queue.put(url)
        
        # Add to finished URLs set
        status_tracker.add_finished_url(url)
        
        logger.info(f"Worker {worker_id}: Successfully processed {url}")
        
    except Exception as e:
        logger.error(f"Worker {worker_id} error processing {url}: {str(e)}")
        
        # Add to retry queue if under retry limit
        if retry_count < MAX_RETRIES:
            logger.info(f"Worker {worker_id}: Will retry URL {url} (attempt {retry_count + 1}/{MAX_RETRIES})")
            await retry_queue.put((url, retry_count + 1))
            status_tracker.increment_retried()
        else:
            logger.warning(f"Worker {worker_id}: URL {url} failed after {MAX_RETRIES} retries")
            status_tracker.increment_processed(False)
            
    finally:
        # Close page and context to free resources
        if page:
            try:
                await page.close()
            except:
                pass
        if context:
            try:
                await context.close()
            except:
                pass
                
        if final_url:
            status_tracker.update_worker(worker_id, url, "Completed", "Success")
            status_tracker.increment_processed(True)
        else:
            status_tracker.update_worker(worker_id, url, "Failed", "Error")

async def url_worker(browser: Browser, worker_id: int, url_queue: asyncio.Queue, retry_queue: asyncio.Queue, output_queue: asyncio.Queue, finished_queue: asyncio.Queue):
    """Worker that processes URLs from the queue"""
    while True:
        try:
            # Check retry queue first
            if not retry_queue.empty():
                url, retry_count = await retry_queue.get()
                logger.info(f"Worker {worker_id}: Processing retry for {url} (attempt {retry_count}/{MAX_RETRIES})")
                await process_url(browser, url, worker_id, output_queue, finished_queue, retry_queue, retry_count)
                retry_queue.task_done()
                continue
                
            # If no retries, get from main queue
            url = await url_queue.get()
            if url is None:  # Termination signal
                url_queue.task_done()
                break
                
            await process_url(browser, url, worker_id, output_queue, finished_queue, retry_queue)
            url_queue.task_done()
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered error: {str(e)}")
            url_queue.task_done()

async def result_writer(output_queue: asyncio.Queue, output_file: str):
    """Worker that writes results to the output file"""
    try:
        # Make sure the directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Opening output file for writing: {output_file}")
        
        # Try async writing first
        try:
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                while True:
                    item = await output_queue.get()
                    
                    if item is None:  # Termination signal
                        logger.info("Result writer received termination signal")
                        output_queue.task_done()
                        break
                        
                    # Write only the URL to the file
                    await f.write(f"{item}\n")
                    await f.flush()  # Ensure it's written to disk
                    
                    output_queue.task_done()
        except Exception as e:
            logger.error(f"Error with aiofiles: {str(e)}. Falling back to standard file IO.")
            
            # Fallback to standard file IO if aiofiles fails
            with open(output_file, 'w', encoding='utf-8') as f:
                while True:
                    item = await output_queue.get()
                    
                    if item is None:  # Termination signal
                        logger.info("Result writer received termination signal")
                        output_queue.task_done()
                        break
                    
                    # Write only the URL to the file
                    f.write(f"{item}\n")
                    f.flush()  # Ensure it's written to disk
                    
                    output_queue.task_done()
                    
    except Exception as e:
        logger.error(f"Fatal error in result writer: {str(e)}")

async def finished_urls_writer(finished_queue: asyncio.Queue, finished_file: str):
    """Worker that writes finished URLs to a separate file"""
    try:
        # Make sure the directory exists
        output_path = Path(finished_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Opening finished URLs file for writing: {finished_file}")
        
        # Try async writing first
        try:
            async with aiofiles.open(finished_file, 'w', encoding='utf-8') as f:
                while True:
                    item = await finished_queue.get()
                    
                    if item is None:  # Termination signal
                        logger.info("Finished URL writer received termination signal")
                        finished_queue.task_done()
                        break
                        
                    # Write the URL to the file
                    await f.write(f"{item}\n")
                    await f.flush()  # Ensure it's written to disk
                    logger.info(f"Added to finished file: {item[:50]}...")
                    
                    finished_queue.task_done()
        except Exception as e:
            logger.error(f"Error with aiofiles for finished URLs: {str(e)}. Falling back to standard file IO.")
            
            # Fallback to standard file IO if aiofiles fails
            with open(finished_file, 'w', encoding='utf-8') as f:
                while True:
                    item = await finished_queue.get()
                    
                    if item is None:  # Termination signal
                        logger.info("Finished URL writer received termination signal")
                        finished_queue.task_done()
                        break
                    
                    # Write the URL to the file
                    f.write(f"{item}\n")
                    f.flush()  # Ensure it's written to disk
                    logger.info(f"Added to finished file (fallback): {item[:50]}...")
                    
                    finished_queue.task_done()
                    
    except Exception as e:
        logger.error(f"Fatal error in finished URL writer: {str(e)}")

async def main():
    # Read input URLs
    try:
        with open(INPUT_FILE, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Could not read input file: {str(e)}")
        return

    total_urls = len(urls)
    logger.info(f"Loaded {total_urls} URLs to process")
    
    # Initialize status tracker
    status_tracker.initialize(total_urls)
    
    # Create queues
    url_queue = asyncio.Queue()
    retry_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    finished_queue = asyncio.Queue()  # New queue for tracking finished URLs
    
    # Add URLs to queue
    for url in urls:
        await url_queue.put(url)
    
    # Launch browser
    async with async_playwright() as playwright:
        # Use persistent context to improve performance
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-dev-shm-usage',  # Helps with memory issues in Docker/Codespaces
                '--disable-gpu',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials'
            ]
        )
        
        # Create workers
        worker_count = min(MAX_WORKERS, total_urls)
        workers = []
        
        for worker_id in range(worker_count):
            status_tracker.update_worker(worker_id, "Initializing", "Idle", "Starting")
            worker = asyncio.create_task(url_worker(browser, worker_id, url_queue, retry_queue, output_queue, finished_queue))
            workers.append(worker)
        
        # Create result writers
        output_writer = asyncio.create_task(result_writer(output_queue, OUTPUT_FILE))
        finished_writer = asyncio.create_task(finished_urls_writer(finished_queue, FINISHED_FILE))
        
        # Create live display
        with Live(status_tracker.generate_table(), refresh_per_second=2) as live:
            # Update the live display every second
            async def update_display():
                while True:
                    live.update(status_tracker.progress)
                    live.update(status_tracker.generate_table())
                    await asyncio.sleep(0.5)
            
            display_task = asyncio.create_task(update_display())
            
            try:
                # Process until both main queue and retry queue are empty
                while not url_queue.empty() or not retry_queue.empty() or sum(not w.done() for w in workers) > 0:
                    await asyncio.sleep(0.5)
                    
                    # If main queue is empty but retry queue isn't, check if workers are idle
                    if url_queue.empty() and not retry_queue.empty():
                        idle_workers = sum(w.done() for w in workers)
                        if idle_workers > 0:
                            logger.info(f"Found {idle_workers} idle workers, redistributing retry queue")
                            # Restart idle workers
                            for i, w in enumerate(workers):
                                if w.done():
                                    workers[i] = asyncio.create_task(
                                        url_worker(browser, i, url_queue, retry_queue, output_queue, finished_queue)
                                    )
                
                # Wait for all workers to finish
                url_queue.join_nowait()  # Just in case there are any items left
                retry_queue.join_nowait()  # Just in case there are any items left
                
                # Send termination signals to workers
                for _ in range(worker_count):
                    await url_queue.put(None)
                
                # Wait for all workers to finish
                await asyncio.gather(*workers)
                
                # Send termination signals to writers
                logger.info("Sending termination signals to writers")
                await output_queue.put(None)
                await finished_queue.put(None)
                
                # Wait for writers to finish
                logger.info("Waiting for writers to finish")
                await output_queue.join()
                await finished_queue.join()
                
            except Exception as e:
                logger.error(f"Error in main processing loop: {str(e)}")
            finally:
                # Cancel display task
                display_task.cancel()
                try:
                    await display_task
                except asyncio.CancelledError:
                    pass
        
        # Close browser
        await browser.close()
    
    # Verify output files exist and have content
    try:
        output_size = os.path.getsize(OUTPUT_FILE)
        logger.info(f"Output file size: {output_size} bytes")
        
        # Count lines in output file
        with open(OUTPUT_FILE, 'r') as f:
            line_count = sum(1 for _ in f)
        logger.info(f"Output file contains {line_count} lines")
        
        # Verify finished URLs file
        finished_size = os.path.getsize(FINISHED_FILE)
        logger.info(f"Finished URLs file size: {finished_size} bytes")
        
        # Count lines in finished file
        with open(FINISHED_FILE, 'r') as f:
            finished_count = sum(1 for _ in f)
        logger.info(f"Finished URLs file contains {finished_count} lines")
        
    except Exception as e:
        logger.error(f"Error checking output files: {str(e)}")
    
    logger.info(f"Processing completed. Processed {status_tracker.processed} URLs.")
    logger.info(f"Successful: {status_tracker.successful}, Failed: {status_tracker.failed}, Retried: {status_tracker.retried}")
    logger.info(f"Results saved to: {OUTPUT_FILE}")
    logger.info(f"Finished URLs saved to: {FINISHED_FILE}")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import aiofiles
        from playwright.async_api import async_playwright
        from rich.console import Console
    except ImportError:
        os.system(f"{sys.executable} -m pip install playwright rich aiofiles")
        os.system(f"{sys.executable} -m playwright install chromium")
        # Re-import after installation
        import aiofiles
        from playwright.async_api import async_playwright
        from rich.console import Console
        from rich.live import Live

    # Run the main function
    asyncio.run(main())
