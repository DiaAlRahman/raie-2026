from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time
import os

BASE_URL = "https://forums.beyondblue.org.au"
START_BOARD_URL = "https://forums.beyondblue.org.au/t5/grief-and-loss/bd-p/c1-sc4-b4"

# =========================
# CHANGE THESE SETTINGS
# =========================

MAX_BOARD_PAGES = 10
# how many forum pages to scrape
# example:
# 1 = only first page
# 5 = first 5 pages
# 10 = first 10 pages

MAX_THREADS_PER_PAGE = 10
# how many threads to scrape from EACH board page
# example:
# 5 = first 5 threads from each page
# 10 = first 10 threads from each page
# set to None if you want all threads from each page

DELAY_BETWEEN_THREADS = 2
# delay in seconds between scraping each thread
# keep this to avoid getting blocked

OUTPUT_FILE = "beyondblue_posts.csv"
# name of the CSV file that will be created

# =========================
# HELPER FUNCTIONS
# =========================

def get_thread_links_from_board(page):
    """
    extract all thread links from the current board page
    a thread link contains '/td-p/'
    """
    html = page.content()
    soup = BeautifulSoup(html, "lxml")

    links = soup.select("a[href]")
    thread_links = []

    for a in links:
        href = a.get("href")

        if href and "/td-p/" in href:
            full_url = urljoin(BASE_URL, href)

            if full_url not in thread_links:
                thread_links.append(full_url)

    return thread_links


def get_next_board_page(soup):
    """
    find the 'next page' link for the board pagination
    returns the full URL of the next board page, or None if not found
    """
    next_link = soup.select_one('a[rel="next"]')

    if next_link and next_link.get("href"):
        return urljoin(BASE_URL, next_link["href"])

    return None


def scrape_thread(page, url):
    """
    open one thread page and extract all post bodies from it
    """
    print(f"\nScraping thread: {url}")

    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    try:
        page.wait_for_selector("div.lia-message-body-content", timeout=30000)
    except:
        print("Could not find posts, skipping this thread.")
        return []

    page.wait_for_timeout(2000)

    html = page.content()
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(strip=True) if soup.title else "No title"
    elements = soup.select("div.lia-message-body-content")

    posts = []

    for i, el in enumerate(elements):
        text = el.get_text("\n", strip=True)

        if text:
            posts.append({
                "board_name": "Grief and loss",     #change to board name
                "thread_title": title,
                "thread_url": url,
                "post_number": i + 1,
                "post_text": text
            })

    print(f"Extracted {len(posts)} posts")
    return posts


# =========================
# MAIN SCRIPT
# =========================

def main():
    all_data = []
    visited_board_pages = set()
    visited_thread_links = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        current_board_url = START_BOARD_URL

        for board_page_number in range(MAX_BOARD_PAGES):
            if not current_board_url:
                print("\nNo more board pages found. Stopping.")
                break

            if current_board_url in visited_board_pages:
                print("\nBoard page already visited. Stopping to avoid duplicates.")
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
            print(f"Found {len(thread_links)} thread links on this board page")

            # =========================
            # CHANGE THREAD LIMIT HERE
            # =========================
            if MAX_THREADS_PER_PAGE is not None:
                thread_links = thread_links[:MAX_THREADS_PER_PAGE]
            # If you want all threads from the page, set:
            # MAX_THREADS_PER_PAGE = None

            print(f"Will scrape {len(thread_links)} thread(s) from this page")

            for link in thread_links:
                if link in visited_thread_links:
                    continue

                visited_thread_links.add(link)

                posts = scrape_thread(page, link)
                all_data.extend(posts)

                time.sleep(DELAY_BETWEEN_THREADS)

            current_board_url = get_next_board_page(soup)

        browser.close()

    if not all_data:
        print("\nNo data collected.")
        return

    df = pd.DataFrame(all_data)

    full_path = os.path.abspath(OUTPUT_FILE)
    df.to_csv(full_path, index=False, encoding="utf-8")

    print(f"\nTotal posts collected: {len(all_data)}")
    print(f"Saved to: {full_path}")


if __name__ == "__main__":
    main()