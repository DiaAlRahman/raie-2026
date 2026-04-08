import requests
from bs4 import BeautifulSoup
import csv
import time

# Target URL for the forum category
base_urls = [
    "https://forums.beyondblue.org.au/t5/suicidal-thoughts-and-self-harm/bd-p/c1-sc2-b4",
    "https://forums.beyondblue.org.au/t5/anxiety/bd-p/c1-sc2-b1",
    "https://forums.beyondblue.org.au/t5/depression/bd-p/c1-sc2-b2",
    "https://forums.beyondblue.org.au/t5/ptsd-and-trauma/bd-p/c1-sc2-b3"
]

# Headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_to_csv(start_page, num_posts, output_file="/workspaces/ProjectRaie/data/raw/beyondblue_posts.csv"):
    current_page = start_page
    post_id = 1
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["post_id", "post"])
        
        while post_id <= num_posts:
            print(f"Scraping page {current_page}...")
            for base_url in base_urls:
                url = f"{base_url}/page/{current_page}"
            
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    # Stop if we hit a 404 or other error
                    if response.status_code != 200:
                        print(f"Stopping: Received status code {response.status_code}")
                        break
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Specific hierarchical selector
                    # div with class 'all-discussions' -> article -> p with class 'body-text'
                    selector = "div.all-discussions article p.body-text"
                    articles = soup.select(selector)
                    
                    if not articles:
                        print("No more posts found using the specified selector.")
                        break
                    
                    for art in articles:
                        content = art.get_text(strip=True)
                        if content:
                            writer.writerow([post_id, content])
                            post_id += 1
                    
                    # Increment page and cooldown to be respectful
                    current_page += 1
                    time.sleep(2) 
                    
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    break

    print(f"Success! Data saved to {output_file}")

if __name__ == "__main__":
    scrape_to_csv(start_page=5, num_posts=40)