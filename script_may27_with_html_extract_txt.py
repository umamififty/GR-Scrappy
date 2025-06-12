#!/usr/bin/env python3
import os
import re
import json
import time
import random
import datetime
import traceback
import logging
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Function to extract readable text from downloaded HTML
def extract_text_from_html(html_path, txt_path):
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove nav, footer, script, and style tags
        for tag in soup(['nav', 'footer', 'script', 'style', 'header', 'noscript']):
            tag.decompose()

        # Extract main visible text
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = '\n'.join(lines)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        return True
    except Exception as e:
        return False

import pandas as pd

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('policy_attachment_downloader')

class PolicyAttachmentDownloader:
    def __init__(self, base_dir, base_url, logger):
        self.base_dir = base_dir
        self.base_url = base_url.rstrip('/')
        self.logger = logger

    def _sanitize_filename(self, name):
        if not isinstance(name, str):
            try:
                name = str(name)
            except Exception:
                return "Unknown"
        if not name or name.lower() in ('nan', 'none'):
            return "Unknown"
        safe = re.sub(r'[\\/*?:"<>|]', '_', name)
        return safe.strip()

    def _make_request_with_backoff(self, url, max_retries=3, timeout=30, stream=False):
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, timeout=timeout, stream=stream)
                resp.raise_for_status()
                return resp
            except Exception as e:
                self.logger.warning(f"Attempt {attempt} failed for URL {url}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
                else:
                    raise

    def _load_checkpoint(self, name):
        path = os.path.join(self.base_dir, f"{name}_checkpoint.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                self.logger.warning(f"Failed to load checkpoint: {path}")
        return None

    def _save_checkpoint(self, name, data):
        path = os.path.join(self.base_dir, f"{name}_checkpoint.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def download_policy_attachments(self, policies_df):
        download_count = 0
        failed_downloads = []
        success_info = []

        if policies_df.empty:
            self.logger.info("No policies found in DataFrame. Exiting.")
            return download_count, failed_downloads, success_info

        cp = self._load_checkpoint('download_attachments') or {}
        processed = set(cp.get('processed', []))
        download_count = cp.get('download_count', 0)
        failed_downloads = cp.get('failures', [])
        success_info = cp.get('success_info', [])

        for idx, row in policies_df.iterrows():
            title = row.get('title')
            url = row.get('url')
            if not title or not url or title in processed:
                continue

            logger.info(f"Processing policy '{title}'")
            category = row.get('category') or 'Uncategorized'
            subcat = row.get('subcategory') or 'General'
            policy_dir = os.path.join(
                self.base_dir,
                self._sanitize_filename(category),
                self._sanitize_filename(subcat),
                self._sanitize_filename(title)
            )
            os.makedirs(policy_dir, exist_ok=True)

            existing_htmls = [f for f in os.listdir(policy_dir) if f.endswith('.html')]
            if existing_htmls:
                self.logger.info(f"Skipping '{title}', HTML attachments already downloaded.")
                processed.add(title)
                continue

            try:
                resp = self._make_request_with_backoff(url, max_retries=3)
                soup = BeautifulSoup(resp.text, 'html.parser')
                links = soup.select(
                    'a.govuk-link.gem-c-attachment__link[href^="/government/publications/"]'
                )
                if not links:
                    logger.info(f"No HTML attachments for '{title}'")
                    processed.add(title)
                    continue

                for link in links:
                    href = link.get('href')
                    if not href:
                        continue
                    full_url = urljoin(self.base_url, href)
                    fname = self._sanitize_filename(link.get_text() or os.path.basename(href)) + '.html'
                    out_path = os.path.join(policy_dir, fname)
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        logger.info(f"Already downloaded: {out_path}")
                        continue

                    logger.info(f"Downloading HTML: {full_url}")
                    fresp = self._make_request_with_backoff(full_url, max_retries=3)
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(fresp.text)
                    logger.info(f"Saved HTML to: {out_path}")
                    download_count += 1

                processed.add(title)

                meta = {
                    'title': title,
                    'url': url,
                    'attachments_downloaded': len(links),
                    'download_count': download_count,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                with open(os.path.join(policy_dir, 'metadata.json'), 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, indent=2)

                self._save_checkpoint('download_attachments', {
                    'processed': list(processed),
                    'download_count': download_count,
                    'failures': failed_downloads,
                    'success_info': success_info
                })

            except Exception as e:
                logger.error(f"Error for '{title}': {e}")
                failed_downloads.append({'title': title, 'error': str(e)})

        return download_count, failed_downloads, success_info

    def extract_plaintext_from_htmls(self):
        self.logger.info("Extracting plain text from downloaded HTML files...")
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.html'):
                    html_path = os.path.join(root, file)
                    txt_path = html_path[:-5] + '.txt'
                    try:
                        with open(html_path, 'r', encoding='utf-8') as hf:
                            soup = BeautifulSoup(hf.read(), 'html.parser')
                            text = soup.get_text(separator='\n', strip=True)
                        with open(txt_path, 'w', encoding='utf-8') as tf:
                            tf.write(text)
                        logger.info(f"Extracted text to: {txt_path}")
                    except Exception as e:
                        logger.error(f"Failed to extract text from {html_path}: {e}")


def main():
    # Use absolute base_dir
    base_dir = '/Documents/Data'
    csv_path = os.path.join(base_dir, 'all_policies.csv')
    if not os.path.exists(csv_path):
        logger.error(f"Missing policy CSV at {csv_path}")
        return
    policies_df = pd.read_csv(csv_path)

    downloader = PolicyAttachmentDownloader(
        base_dir=base_dir,
        base_url='https://www.gov.uk',
        logger=logger
    )

    downloader.download_policy_attachments(policies_df)
    downloader.extract_plaintext_from_htmls()


if __name__ == '__main__':
    main()
