from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time
import os

BASE_URL = "https://forums.beyondblue.org.au"
START_BOARD_URL = "https://forums.beyondblue.org.au/t5/suicidal-thoughts-and-self-harm/bd-p/c1-sc2-b4/page/5" # link of the start page



# =========================
# SETTINGS
# =========================

MAX_BOARD_PAGES = 2
MAX_THREADS_PER_PAGE = None  # None = all posts in the page
DELAY_BETWEEN_THREADS = 2

OUTPUT_FILE = "beyondblue_posts.csv" # output file name

# To run scraper in dev container, do 
# xvfb-run -a python3 src/scraper/playwright_scraper.py 

# =========================
# HELPER FUNCTIONS
# =========================

def get_thread_links_from_board(page):

    html = page.content()
    soup = BeautifulSoup(html, "lxml")

    links = soup.select("a[href]")
    thread_links = []

    for a in links:
        href = a.get("href")

        if href and "/td-p/" in href:      # filter thread links
            full_url = urljoin(BASE_URL, href)

            if full_url not in thread_links:        # avoid duplicates
                thread_links.append(full_url)

    return thread_links


def get_next_board_page(soup):
    next_link = soup.select_one('a[rel="next"]')     # pagination

    if next_link and next_link.get("href"):
        return urljoin(BASE_URL, next_link["href"])

    return None


def scrape_post(page, url):
    """
    open one thread page and extract ONLY the first post
    """
    print(f"\nScraping thread: {url}")
    page.screenshot(path="out.png")
    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    try:
        page.wait_for_selector("div.lia-message-body-content", timeout=30000)    # wait for post
    except:
        print("Could not find posts, skipping this thread.")
        return None

    page.wait_for_timeout(2000)      # small buffer

    html = page.content()
    soup = BeautifulSoup(html, "lxml")

    first_post = soup.select_one("div.lia-message-body-content")     # first post only

    if not first_post:
        print("No first post found, skipping.")
        return None

    text = first_post.get_text("\n", strip=True)

    if not text:
        print("Empty post, skipping.")
        return None

    print("Extracted first post")
    return text


# =========================
# MAIN SCRIPT
# =========================

def main():
    all_data = []
    visited_board_pages = set()
    visited_thread_links = set()
    post_counter = 1

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        current_board_url = START_BOARD_URL

        for board_page_number in range(MAX_BOARD_PAGES):
            if not current_board_url:
                break

            if current_board_url in visited_board_pages:
                break

            print(f"\n{'='*80}")
            print(f"Opening board page {board_page_number + 1}: {current_board_url}")
            print(f"{'='*80}")

            visited_board_pages.add(current_board_url)

            page.goto(current_board_url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2000)

            html = page.content()
            soup = BeautifulSoup(html, "lxml")

            thread_links = get_thread_links_from_board(page)

            print(f"Found {len(thread_links)} threads")

            if MAX_THREADS_PER_PAGE is not None:
                thread_links = thread_links[:MAX_THREADS_PER_PAGE]

            for link in thread_links:
                if link in visited_thread_links:
                    continue

                visited_thread_links.add(link)

                raw_post_text = scrape_post(page, link)
                sanitised_post_text = raw_post_text.replace("\n", "")
                if raw_post_text:
                    all_data.append({
                        "post_id": post_counter,
                        "post": sanitised_post_text
                    })
                    post_counter += 1

                time.sleep(DELAY_BETWEEN_THREADS)

            current_board_url = get_next_board_page(soup)

        browser.close()

    if not all_data:
        print("\nNo data collected.")
        return

    df = pd.DataFrame(all_data)
    # Create data/raw directory if it does not exist
    try:
        os.makedirs("data/raw")
        os.chmod("data/raw", 0o777)
        print(f"Directory 'data/raw' created successfully.")
    except FileExistsError:
        print(f"Directory 'data/raw' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create directory 'data/raw'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    output_path = "data/raw/" + OUTPUT_FILE
    full_path = os.path.abspath(output_path)
    df.to_csv(full_path, index=False, encoding="utf-8")

    print(f"\nTotal posts collected: {len(all_data)}")
    print(f"Saved to: {full_path}")


if __name__ == "__main__":
    main()