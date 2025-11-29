import os
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin, urlparse
import time

# Configuration
BASE_URL = "https://docs.browser.vision/"
OUTPUT_DIR = "scraped_docs"

def get_soup(url):
    """Fetches a URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def save_as_markdown(url, content, title):
    """Saves the content as a Markdown file."""
    if not content:
        return

    # Create a filename from the URL path
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")
    if not path:
        filename = "index.md"
    else:
        filename = path.replace("/", "_") + ".md"
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    markdown_content = f"# {title}\n\nSource: {url}\n\n{md(str(content))}"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Saved: {filepath}")

def scrape_docs():
    """Main function to scrape the documentation."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Starting scrape of {BASE_URL}...")
    soup = get_soup(BASE_URL)
    if not soup:
        return

    # Find all links in the sidebar/navigation
    # Based on exploration, it seems to be a Nextra site, usually nav is in <aside> or specific classes
    # Let's try to find all internal links first and filter them
    
    # Strategy: 
    # 1. Find all links on the homepage that start with the BASE_URL or are relative.
    # 2. Filter for those that look like documentation pages.
    # 3. Visit each unique link.
    
    visited = set()
    to_visit = [BASE_URL]
    
    # We'll use a set to keep track of unique pages to scrape
    # Since we don't have a perfect sitemap, we'll do a simple crawl of the links found on the homepage 
    # and maybe one level deep if needed, but usually the sidebar has everything.
    
    # Let's grab the sidebar links specifically if possible. 
    # Common Nextra sidebar selector might be 'aside' or 'nav'
    # If not sure, we can just grab all links from the homepage that are within the domain.
    
    links_to_scrape = set()
    
    # Try to find sidebar links first for better structure
    sidebar = soup.find('aside') or soup.find('nav')
    if sidebar:
        print("Found sidebar/nav, extracting links...")
        for a in sidebar.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(BASE_URL, href)
            if full_url.startswith(BASE_URL):
                links_to_scrape.add(full_url)
    else:
        print("Sidebar not found, extracting all internal links from homepage...")
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(BASE_URL, href)
            if full_url.startswith(BASE_URL):
                links_to_scrape.add(full_url)
    
    # Add homepage if not present
    links_to_scrape.add(BASE_URL)
    
    print(f"Found {len(links_to_scrape)} unique pages to scrape.")
    
    for url in sorted(list(links_to_scrape)):
        if url in visited:
            continue
            
        print(f"Scraping: {url}")
        page_soup = get_soup(url)
        if not page_soup:
            continue
            
        # Extract main content
        # Nextra usually puts content in <main> or <article>
        main_content = page_soup.find('main') or page_soup.find('article')
        
        if not main_content:
            print(f"Could not find main content for {url}, skipping.")
            # Fallback: try to use the body but remove nav/header/footer
            # This is risky, better to just log it for now.
            continue
            
        # Get title
        title = page_soup.title.string if page_soup.title else "Untitled"
        
        save_as_markdown(url, main_content, title)
        visited.add(url)
        
        # Be polite
        time.sleep(0.5)

if __name__ == "__main__":
    scrape_docs()
