import requests
import time
import os
import re
import traceback
import json
import logging
import datetime
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
from collections import Counter

class GovUKPolicyScraper:
    def __init__(self, base_dir="/Volumes/Research/policy_data", log_level=logging.INFO):
        """
        Initialize the scraper with improved logging and failsafe mechanisms
        """
        # Clean existing base directory for fresh runs
        if os.path.exists(base_dir):
            import shutil
            shutil.rmtree(base_dir)

        self.base_url = "https://www.gov.uk"
        self.base_dir = base_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolicyResearch/1.0 (Academic Research)'
        })
        
        # Create base directory and logs directory
        base_dir = "/Volumes/Research"
        os.makedirs(self.base_dir, exist_ok=True)
        logs_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging(logs_dir, log_level)
        
        # Checkpoint directory
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.logger.info(f"Initialized scraper with base directory: {self.base_dir}")
    
    def _setup_logging(self, logs_dir, log_level):
        """Setup logging with file and console handlers"""
        logger = logging.getLogger('policy_scraper')
        logger.setLevel(log_level)
        
        # Create a unique log file for this run
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(logs_dir, f'scraper_{timestamp}.log')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def search_policies(self, search_term=None, page_limit=99999, exclude_types=None, resume_from_page=1):
        """
        Search for policy papers on gov.uk with failsafe mechanisms
        
        Args:
            search_term (str): Optional keyword to search for
            page_limit (int): Maximum number of pages to scrape
            exclude_types (list): Types of documents to exclude, e.g. ['consultations']
            resume_from_page (int): Page to resume from (for recovery)
            
        Returns:
            pd.DataFrame: DataFrame containing search results
        """
        all_results = []
        
        # Check for existing checkpoint to resume
        checkpoint_data = self._load_checkpoint('search_policies')
        if checkpoint_data and resume_from_page == 1:
            self.logger.info(f"Found checkpoint with {len(checkpoint_data['results'])} existing results")
            all_results = checkpoint_data['results']
            resume_from_page = checkpoint_data['next_page']
        
        # Use only policy-papers, not consultations
        search_url = f"{self.base_url}/search/policy-papers-and-consultations"
        
        params = {
            'keywords': search_term if search_term else '',
            'content_store_document_type': 'policy_papers'
        }
        
        self.logger.info(f"Starting full extraction process for all policy papers from page {resume_from_page}")
        self.logger.info(f"Scraping all policy papers (up to {page_limit} pages)")
        
        for page in range(resume_from_page, page_limit + 1):
            try:
                params['page'] = page
                self.logger.info(f"Scraping page {page}...")
                
                # Use backoff strategy for requests
                try:
                    response = self._make_request_with_backoff(search_url, params=params)
                except Exception as e:
                    self.logger.error(f"Failed to get page {page} after multiple retries: {e}")
                    # Save checkpoint before skipping
                    self._save_checkpoint('search_policies', {'results': all_results, 'next_page': page})
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                results_container = soup.find('div', id='js-results')
                
                if not results_container:
                    self.logger.warning(f"Could not find results container with id 'js-results' on page {page}")
                    continue
                    
                results_list = results_container.find_all('li', class_='gem-c-document-list__item')
                
                if not results_list:
                    self.logger.info(f"No results found on page {page}, stopping pagination")
                    break
                
                self.logger.info(f"Found {len(results_list)} results on page {page}")
                
                page_results = []
                for idx, item in enumerate(results_list):
                    try:
                        # Skip consultations if in exclude_types
                        if exclude_types and any(excluded in item.text.lower() for excluded in exclude_types):
                            continue
                            
                        policy_data = self._extract_policy_data(item)
                        if policy_data:
                            # Extract categories
                            policy_url = policy_data['url']
                            try:
                                categories = self._extract_categories(policy_url)
                                policy_data.update(categories)
                            except Exception as e:
                                self.logger.warning(f"Error extracting categories for {policy_data['title']}: {e}")
                                policy_data.update({'category': None, 'subcategory': None})
                            
                            page_results.append(policy_data)
                            self.logger.info(f"Extracted: {policy_data['title']}")
                        
                        # Save checkpoint after every 5 items
                        if idx > 0 and idx % 5 == 0:
                            temp_results = all_results + page_results
                            self._save_checkpoint('search_policies', {'results': temp_results, 'next_page': page})
                            self.logger.debug(f"Saved intermediate checkpoint with {len(temp_results)} results")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing item on page {page}: {e}")
                        self.logger.debug(f"Traceback: {traceback.format_exc()}")
                        continue
                
                all_results.extend(page_results)
                self.logger.info(f"Total results so far: {len(all_results)}")
                
                # Save checkpoint after each page
                self._save_checkpoint('search_policies', {'results': all_results, 'next_page': page + 1})
                
                # Add a delay with jitter to be respectful to the server
                sleep_time = 2 + random.random()
                self.logger.debug(f"Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                # Save checkpoint before continuing to next page
                self._save_checkpoint('search_policies', {'results': all_results, 'next_page': page + 1})
        
        policies_df = pd.DataFrame(all_results)
        self.logger.info(f"Found {len(policies_df)} policies")
        
        # Save the policies data
        if not policies_df.empty:
            csv_path = os.path.join(self.base_dir, "all_policies.csv")
            try:
                policies_df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved policies data to {csv_path}")
            except Exception as e:
                self.logger.error(f"Error saving policies data to CSV: {e}")
                # Backup save as JSON
                json_path = os.path.join(self.base_dir, "all_policies.json")
                try:
                    policies_df.to_json(json_path, orient='records', indent=2)
                    self.logger.info(f"Saved backup policies data to {json_path}")
                except Exception as json_e:
                    self.logger.error(f"Error saving backup JSON: {json_e}")
        
        return policies_df
    
    def _extract_policy_data(self, item):
        """
        Extract policy data from a search result item with error handling
        
        Args:
            item: BeautifulSoup object for a search result
            
        Returns:
            dict: Dictionary with policy data or None on failure
        """
        try:
            title_tag = item.find('a', class_='gem-c-document-list__item-title')
            if not title_tag:
                # Try alternative selector
                title_tag = item.select_one('div.gem-c-document-list__item-title a')
                if not title_tag:
                    self.logger.warning("Could not find title tag in search result item")
                    return None
                    
            title = title_tag.text.strip()
            link = title_tag.get('href')
            full_link = self.base_url + link if link.startswith('/') else link
            
            # Extract metadata
            metadata_container = item.find('ul', class_='gem-c-document-list__item-metadata')
            metadata = {
                'published_date': None,
                'updated_date': None,
                'department': None,
                'type': None
            }
            
            if metadata_container:
                metadata_items = metadata_container.find_all('li')
                for meta_item in metadata_items:
                    text = meta_item.text.strip()
                    if "Published" in text:
                        date_part = text.replace("Published: ", "").strip()
                        metadata['published_date'] = date_part
                    elif "Organisation" in text or "Department" in text or "From" in text:
                        metadata['department'] = text.replace("Organisation: ", "").replace("Department: ", "").replace("From: ", "")
                    else:
                        metadata['type'] = text
            
            # If dates weren't found in the list metadata, try to get them from the policy page metadata
            if (not metadata['published_date'] or not metadata['updated_date']) and full_link:
                try:
                    # Get the policy detail page with error handling
                    response = self._make_request_with_backoff(full_link, max_retries=3)
                    
                    detail_soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for the published date in the head metadata
                    published_meta = detail_soup.find('meta', attrs={'name': 'govuk:first-published-at'})
                    if published_meta:
                        published_date = published_meta.get('content')
                        if published_date:
                            # Convert ISO format to more readable format (optional)
                            try:
                                dt = datetime.datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                                metadata['published_date'] = dt.strftime('%d %B %Y')
                            except Exception as e:
                                # If date parsing fails, just use the original string
                                self.logger.debug(f"Error parsing published date: {e}")
                                metadata['published_date'] = published_date
                    
                    # Look for the updated date in the head metadata
                    updated_meta = detail_soup.find('meta', attrs={'name': 'govuk:updated-at'})
                    if updated_meta:
                        updated_date = updated_meta.get('content')
                        if updated_date:
                            # Convert ISO format to more readable format (optional)
                            try:
                                dt = datetime.datetime.fromisoformat(updated_date.replace('Z', '+00:00'))
                                metadata['updated_date'] = dt.strftime('%d %B %Y')
                            except Exception as e:
                                # If date parsing fails, just use the original string
                                self.logger.debug(f"Error parsing updated date: {e}")
                                metadata['updated_date'] = updated_date
                    
                    # If still no published date, try the public web published date
                    if not metadata['published_date']:
                        public_meta = detail_soup.find('meta', attrs={'name': 'govuk:public-updated-at'})
                        if public_meta:
                            public_date = public_meta.get('content')
                            if public_date:
                                try:
                                    dt = datetime.datetime.fromisoformat(public_date.replace('Z', '+00:00'))
                                    metadata['published_date'] = dt.strftime('%d %B %Y')
                                except Exception as e:
                                    self.logger.debug(f"Error parsing public date: {e}")
                                    metadata['published_date'] = public_date
                
                except Exception as e:
                    self.logger.warning(f"Error fetching policy page for date extraction: {e}")
            
            # Extract description
            description_tag = item.find('p', class_='gem-c-document-list__item-description')
            description = description_tag.text.strip() if description_tag else None
            
            return {
                'title': title,
                'url': full_link,
                'description': description,
                **metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting policy data: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_categories(self, url):
        """
        Extract categories from a policy page with error handling
        
        Args:
            url (str): URL of the policy page
            
        Returns:
            dict: Dictionary with category information
        """
        try:
            response = self._make_request_with_backoff(url, max_retries=3)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            categories = {
                'category': None,
                'subcategory': None
            }
            
            # Try to find categories from breadcrumbs
            breadcrumb = soup.find('nav', class_='govuk-breadcrumbs')
            if breadcrumb:
                breadcrumb_items = breadcrumb.find_all('li', class_='govuk-breadcrumbs__list-item')
                
                # Breadcrumb structure is typically: Home > Category > Subcategory
                if len(breadcrumb_items) >= 2:
                    # Skip "Home", get category
                    category_item = breadcrumb_items[1]
                    category_link = category_item.find('a')
                    if category_link:
                        categories['category'] = category_link.text.strip()
                
                # Get subcategory if available
                if len(breadcrumb_items) >= 3:
                    subcategory_item = breadcrumb_items[2]
                    subcategory_link = subcategory_item.find('a')
                    if subcategory_link:
                        categories['subcategory'] = subcategory_link.text.strip()
            
            # If not found in breadcrumbs, try alternative methods
            if not categories['category']:
                # Look for topic tags
                topic_tags = soup.find_all('a', class_=lambda c: c and 'topic' in (c or '').lower())
                if topic_tags:
                    categories['category'] = topic_tags[0].text.strip()
            
            return categories
            
        except Exception as e:
            self.logger.error(f"Error extracting categories from {url}: {e}")
            # Return empty categories rather than failing
            return {'category': None, 'subcategory': None}
    
    def download_policy_attachments(self, policies_df):
        """
        Download and organize attachments from policies with comprehensive error handling
        
        Args:
            policies_df (pd.DataFrame): DataFrame with policy data
            
        Returns:
            tuple: (download_count, failed_downloads, success_info)
        """
        download_count = 0
        failed_downloads = []
        success_info = []
        
        if policies_df.empty:
            self.logger.warning("No policies to process for attachment download")
            return download_count, failed_downloads, success_info
        
        self.logger.info(f"Downloading attachments for {len(policies_df)} policies...")
        
        # Check for existing checkpoint to resume
        checkpoint_data = self._load_checkpoint('download_attachments')
        processed_policies = set()
        
        if checkpoint_data:
            self.logger.info(f"Found download checkpoint with {len(checkpoint_data['processed'])} processed policies")
            processed_policies = set(checkpoint_data['processed'])
            download_count = checkpoint_data.get('download_count', 0)
            failed_downloads = checkpoint_data.get('failures', [])
            success_info = checkpoint_data.get('success_info', [])
        
        # Process each policy
        for idx, policy in policies_df.iterrows():
            try:
                title = policy['title']
                url = policy['url']
                
                # Skip already processed policies
                if title in processed_policies:
                    self.logger.info(f"Skipping already processed policy: {title}")
                    continue
                
                category = policy.get('category', 'Uncategorized')
                subcategory = policy.get('subcategory', 'General')
                
                self.logger.info(f"Processing: {title}")
                
                # Create folder structure
                category_dir = self._sanitize_filename(category)
                subcategory_dir = self._sanitize_filename(subcategory)
                policy_dir = self._sanitize_filename(title)
                
                full_path = os.path.join(self.base_dir, category_dir, subcategory_dir, policy_dir)
                os.makedirs(full_path, exist_ok=True)
                
                policy_attachments = []
                url_processed = False
                
                try:
                    # Get policy detail page
                    response = self._make_request_with_backoff(url, max_retries=3)
                    url_processed = True
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find attachments
                    attachment_sections = soup.find_all(['div', 'section'], class_=lambda c: c and 'attachment' in c)
                    if not attachment_sections:
                        self.logger.info(f"No attachments found for: {title}")
                        # Mark as processed even if no attachments
                        processed_policies.add(title)
                        success_info.append({
                            'policy': title,
                            'attachments_found': 0,
                            'attachments_downloaded': 0,
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                        # Save checkpoint periodically
                        if idx % 5 == 0:
                            self._save_checkpoint('download_attachments', {
                                'processed': list(processed_policies),
                                'download_count': download_count,
                                'failures': failed_downloads,
                                'success_info': success_info
                            })
                        continue
                    
                    # Extract and download attachments
                    total_attachments = 0
                    successful_downloads = 0
                    
                    for section in attachment_sections:
                        attachment_links = section.find_all('a')
                        for link in attachment_links:
                            try:
                                href = link.get('href')
                                if not href:
                                    continue
                                    
                                # Only download document files
                                if re.search(r'\.(pdf|doc|docx|xls|xlsx|ppt|pptx|csv)$', href, re.I):
                                    total_attachments += 1
                                    attachment_url = urljoin(self.base_url, href)
                                    filename = os.path.basename(href)
                                    save_path = os.path.join(full_path, filename)
                                    
                                    # Check if file already exists and has content
                                    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                                        self.logger.info(f"File already exists: {filename}")
                                        successful_downloads += 1
                                        policy_attachments.append({
                                            'filename': filename,
                                            'url': attachment_url,
                                            'path': save_path,
                                            'size': os.path.getsize(save_path)
                                        })
                                        continue
                                    
                                    # Download the file
                                    self.logger.info(f"Downloading: {filename}")
                                    try:
                                        file_response = self._make_request_with_backoff(
                                            attachment_url, 
                                            stream=True, 
                                            max_retries=3, 
                                            timeout=60
                                        )
                                        
                                        with open(save_path, 'wb') as f:
                                            for chunk in file_response.iter_content(chunk_size=8192):
                                                f.write(chunk)
                                        
                                        file_size = os.path.getsize(save_path)
                                        
                                        # Verify file was downloaded properly
                                        if file_size == 0:
                                            raise Exception("Downloaded file is empty")
                                        
                                        policy_attachments.append({
                                            'filename': filename,
                                            'url': attachment_url,
                                            'path': save_path,
                                            'size': file_size
                                        })
                                        
                                        download_count += 1
                                        successful_downloads += 1
                                        self.logger.info(f"Saved to: {save_path}")
                                        
                                    except Exception as e:
                                        self.logger.error(f"Error downloading {filename}: {e}")
                                        failed_downloads.append({
                                            'policy': title,
                                            'file': filename,
                                            'url': attachment_url,
                                            'error': str(e),
                                            'timestamp': datetime.datetime.now().isoformat()
                                        })
                                    
                                    # Be nice to the server
                                    time.sleep(1 + random.random())
                            
                            except Exception as e:
                                self.logger.error(f"Error processing attachment link: {e}")
                                continue
                    
                    # Create metadata file with policy information
                    metadata = {
                        'title': title,
                        'url': url,
                        'published_date': policy.get('published_date'),
                        'updated_date': policy.get('updated_date'),
                        'department': policy.get('department'),
                        'category': category,
                        'subcategory': subcategory,
                        'description': policy.get('description'),
                        'attachments': policy_attachments,
                        'total_attachments_found': total_attachments,
                        'successful_downloads': successful_downloads,
                        'scraped_at': datetime.datetime.now().isoformat()
                    }
                    
                    metadata_path = os.path.join(full_path, 'metadata.json')
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    # Record success info
                    success_info.append({
                        'policy': title,
                        'attachments_found': total_attachments,
                        'attachments_downloaded': successful_downloads,
                        'path': full_path,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    
                    # Mark as processed
                    processed_policies.add(title)
                    
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Error fetching policy page {title}: {e}")
                    if not url_processed:
                        failed_downloads.append({
                            'policy': title,
                            'file': 'policy_page',
                            'url': url,
                            'error': str(e),
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                
                # Save checkpoint after each policy
                if idx % 5 == 0 or idx == len(policies_df) - 1:
                    self._save_checkpoint('download_attachments', {
                        'processed': list(processed_policies),
                        'download_count': download_count,
                        'failures': failed_downloads,
                        'success_info': success_info
                    })
                
            except Exception as e:
                self.logger.error(f"Error processing policy {policy.get('title', 'Unknown')}: {e}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                # Continue to next policy
                continue
        
        self.logger.info(f"Total attachments downloaded: {download_count}")
        self.logger.info(f"Total failed downloads: {len(failed_downloads)}")
        
        # Save failed downloads for later retry
        if failed_downloads:
            failed_path = os.path.join(self.base_dir, 'failed_downloads.json')
            try:
                with open(failed_path, 'w', encoding='utf-8') as f:
                    json.dump(failed_downloads, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Failed downloads saved to {failed_path}")
            except Exception as e:
                self.logger.error(f"Error saving failed downloads: {e}")
        
        # Save final download summary
        download_summary = {
            'total_policies': len(policies_df),
            'policies_processed': len(processed_policies),
            'total_attachments_downloaded': download_count,
            'failed_downloads': len(failed_downloads),
            'completion_time': datetime.datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.base_dir, 'download_summary.json')
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(download_summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Download summary saved to {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving download summary: {e}")
        
        return download_count, failed_downloads, success_info
    
    def retry_failed_downloads(self, failed_downloads_path=None, max_retries=3):
        """
        Retry previously failed downloads
        
        Args:
            failed_downloads_path (str): Path to the JSON file with failed downloads
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            tuple: (successful_retries, remaining_failures)
        """
        if failed_downloads_path is None:
            failed_downloads_path = os.path.join(self.base_dir, 'failed_downloads.json')
        
        if not os.path.exists(failed_downloads_path):
            self.logger.warning(f"Failed downloads file not found: {failed_downloads_path}")
            return 0, []
        
        try:
            with open(failed_downloads_path, 'r', encoding='utf-8') as f:
                failed_downloads = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading failed downloads file: {e}")
            return 0, []
        
        self.logger.info(f"Retrying {len(failed_downloads)} failed downloads...")
        
        successful_retries = 0
        remaining_failures = []
        
        # Create retry directory
        retry_dir = os.path.join(self.base_dir, 'retried_downloads')
        os.makedirs(retry_dir, exist_ok=True)
        
        for i, item in enumerate(failed_downloads):
            policy = item.get('policy', 'Unknown')
            filename = item.get('file', 'Unknown')
            url = item.get('url', '')
            
            if not url:
                self.logger.warning(f"Skipping item with no URL: {item}")
                remaining_failures.append(item)
                continue
            
            self.logger.info(f"Retrying download for {policy} - {filename} ({i+1}/{len(failed_downloads)})")
            
            for attempt in range(max_retries):
                try:
                    if filename == 'policy_page':
                        # Handle policy page retry logic
                        response = self._make_request_with_backoff(url, timeout=30, max_retries=2)
                        self.logger.info(f"Successfully fetched policy page for {policy}")
                        successful_retries += 1
                        break
                    else:
                        # Handle file download retry logic
                        policy_dir = os.path.join(retry_dir, self._sanitize_filename(policy))
                        os.makedirs(policy_dir, exist_ok=True)
                        
                        file_path = os.path.join(policy_dir, filename)
                        
                        file_response = self._make_request_with_backoff(
                            url, 
                            stream=True, 
                            timeout=60,
                            max_retries=2
                        )
                        
                        with open(file_path, 'wb') as f:
                            for chunk in file_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # Verify file was downloaded properly
                        if os.path.getsize(file_path) == 0:
                            raise Exception("Downloaded file is empty")
                        
                        self.logger.info(f"Successfully downloaded {filename} for {policy}")
                        successful_retries += 1
                        break
                except Exception as e:
                    self.logger.warning(f"Retry {attempt+1}/{max_retries} failed for {filename}: {e}")
                    if attempt == max_retries - 1:
                        item['error'] = f"Failed after {max_retries} retries: {str(e)}"
                        item['last_retry_time'] = datetime.datetime.now().isoformat()
                        remaining_failures.append(item)
                
                # Wait longer between retries with exponential backoff
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
            
            # Save checkpoint every 5 items
            if i % 5 == 0 and i > 0:
                checkpoint_data = {
                    'successful_retries': successful_retries,
                    'remaining_failures': remaining_failures,
                    'items_processed': i + 1,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                self._save_checkpoint('retry_downloads', checkpoint_data)
        
        # Save remaining failures
        if remaining_failures:
            remaining_path = os.path.join(self.base_dir, 'remaining_failures.json')
            try:
                with open(remaining_path, 'w', encoding='utf-8') as f:
                    json.dump(remaining_failures, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Remaining failures saved to {remaining_path}")
            except Exception as e:
                self.logger.error(f"Error saving remaining failures: {e}")
        
        # Save retry summary
        retry_summary = {
            'original_failures': len(failed_downloads),
            'successful_retries': successful_retries,
            'remaining_failures': len(remaining_failures),
            'completion_time': datetime.datetime.now().isoformat()
        }
        
        retry_summary_path = os.path.join(self.base_dir, 'retry_summary.json')
        try:
            with open(retry_summary_path, 'w', encoding='utf-8') as f:
                json.dump(retry_summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Retry summary saved to {retry_summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving retry summary: {e}")
        
        return successful_retries, remaining_failures
    
    def categorize_policies(self, policies_df):
        """
        Organize policies by category and subcategory with failsafe mechanisms
        
        Args:
            policies_df (pd.DataFrame): DataFrame with policy data
            
        Returns:
            dict: Dictionary with categorized policies
        """
        if policies_df.empty:
            self.logger.warning("No policies to categorize")
            return {}
        
        self.logger.info(f"Categorizing {len(policies_df)} policies...")
        
        # Initialize category structure
        categories = {}
        
        # Check for existing checkpoint
        checkpoint_data = self._load_checkpoint('categorize_policies')
        if checkpoint_data:
            self.logger.info("Found categorization checkpoint, resuming from there")
            categories = checkpoint_data['categories']
        
        try:
            # Group by category and subcategory
            for i, (_, policy) in enumerate(policies_df.iterrows()):
                try:
                    category = policy.get('category', 'Uncategorized')
                    subcategory = policy.get('subcategory', 'General')
                    
                    if category not in categories:
                        categories[category] = {'subcategories': {}}
                    
                    if subcategory not in categories[category]['subcategories']:
                        categories[category]['subcategories'][subcategory] = {'policies': []}
                    
                    # Add policy to appropriate subcategory
                    categories[category]['subcategories'][subcategory]['policies'].append({
                        'title': policy['title'],
                        'url': policy['url'],
                        'published_date': policy.get('published_date'),
                        'department': policy.get('department')
                    })
                    
                    # Save checkpoint periodically
                    if i > 0 and i % 100 == 0:
                        self._save_checkpoint('categorize_policies', {'categories': categories})
                        self.logger.debug(f"Saved categorization checkpoint at {i} policies")
                        
                except Exception as e:
                    self.logger.error(f"Error categorizing policy {policy.get('title', 'Unknown')}: {e}")
                    continue
            
            # Calculate counts
            for category, cat_data in categories.items():
                cat_policy_count = 0
                
                for subcategory, subcat_data in cat_data['subcategories'].items():
                    policies_count = len(subcat_data['policies'])
                    cat_data['subcategories'][subcategory]['count'] = policies_count
                    cat_policy_count += policies_count
                
                cat_data['count'] = cat_policy_count
            
            # Save category structure
            categories_path = os.path.join(self.base_dir, 'category_structure.json')
            with open(categories_path, 'w', encoding='utf-8') as f:
                json.dump(categories, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Category structure saved to {categories_path}")
            
            # Print category summary
            self.logger.info("\nCategory Summary:")
            self.logger.info("=" * 40)
            for category, cat_data in categories.items():
                self.logger.info(f"{category}: {cat_data['count']} policies")
                for subcategory, subcat_data in cat_data['subcategories'].items():
                    self.logger.info(f"  - {subcategory}: {subcat_data['count']} policies")
            
            return categories
            
        except Exception as e:
            self.logger.error(f"Error in categorize_policies: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Save partial categorization
            if categories:
                self._save_checkpoint('categorize_policies', {'categories': categories})
                partial_path = os.path.join(self.base_dir, 'partial_categories.json')
                with open(partial_path, 'w', encoding='utf-8') as f:
                    json.dump(categories, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Partial category structure saved to {partial_path}")
            
            return categories
    
    def _make_request_with_backoff(self, url, method='get', max_retries=5, **kwargs):
        """
        Make a request with exponential backoff for handling rate limits and server issues
        
        Args:
            url (str): URL to request
            method (str): HTTP method (get, post, etc.)
            max_retries (int): Maximum number of retry attempts
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response: Response object
        """
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 30  # Default timeout
        
        for attempt in range(max_retries):
            try:
                if method.lower() == 'get':
                    response = self.session.get(url, **kwargs)
                elif method.lower() == 'post':
                    response = self.session.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response
                
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                # Calculate backoff time with jitter
                backoff_time = (2 ** attempt) + random.uniform(0, 1)
                
                # Don't retry on client errors (except 429 Too Many Requests)
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if 400 <= status_code < 500 and status_code != 429:
                        self.logger.warning(f"Client error {status_code}, not retrying: {url}")
                        raise
                
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
                    self.logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                else:
                    self.logger.error(f"Request failed after {max_retries} attempts: {url}")
                    raise
    
    def _save_checkpoint(self, operation, data):
        """
        Save checkpoint to resume operations later
        
        Args:
            operation (str): Name of the operation
            data (dict): Checkpoint data
        """
        try:
            # Add timestamp
            checkpoint = {
                'timestamp': datetime.datetime.now().isoformat(),
                'operation': operation,
                'data': data
            }
            
            # Use timestamped checkpoints to avoid corruption
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{operation}_{timestamp}.json')
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            
            # Create a symbolic link or copy to latest
            latest_path = os.path.join(self.checkpoint_dir, f'{operation}_latest.json')
            if os.path.exists(latest_path):
                os.remove(latest_path)
            
            # Copy the file to latest
            import shutil
            shutil.copy2(checkpoint_path, latest_path)
            
            self.logger.debug(f"Checkpoint saved for {operation}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint for {operation}: {e}")
    
    def _load_checkpoint(self, operation):
        """
        Load checkpoint to resume operations
        
        Args:
            operation (str): Name of the operation
            
        Returns:
            dict or None: Checkpoint data if exists, otherwise None
        """
        latest_path = os.path.join(self.checkpoint_dir, f'{operation}_latest.json')
        
        if not os.path.exists(latest_path):
            return None
        
        try:
            with open(latest_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            self.logger.info(f"Loaded checkpoint for {operation} from {checkpoint.get('timestamp', 'unknown time')}")
            return checkpoint.get('data', {})
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint for {operation}: {e}")
            
            # Try to find the most recent valid checkpoint
            try:
                import glob
                checkpoints = glob.glob(os.path.join(self.checkpoint_dir, f'{operation}_*.json'))
                # Filter out the latest
                checkpoints = [c for c in checkpoints if not c.endswith('_latest.json')]
                
                if checkpoints:
                    # Sort by modification time, newest first
                    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    
                    for backup_path in checkpoints[:3]:  # Try the 3 most recent
                        try:
                            with open(backup_path, 'r', encoding='utf-8') as f:
                                backup_checkpoint = json.load(f)
                            
                            self.logger.warning(f"Using backup checkpoint from {backup_path}")
                            return backup_checkpoint.get('data', {})
                        except:
                            continue
            except Exception as backup_error:
                self.logger.error(f"Error finding backup checkpoints: {backup_error}")
            
            return None
    
    def _sanitize_filename(self, name):
        """
        Convert a string to a valid directory name
        
        Args:
            name (str): String to sanitize
            
        Returns:
            str: Sanitized string
        """
        if not name:
            return "Unknown"
            
        # Replace invalid characters
        s = re.sub(r'[\\/*?:"<>|]', '', name)
        # Replace multiple spaces with a single space
        s = re.sub(r'\s+', ' ', s)
        # Trim the name if it's too long
        if len(s) > 75:
            s = s[:75]
        
        return s
    
    def generate_scrape_report(self):
        """
        Generate a comprehensive report of the scraping process
        
        Returns:
            dict: Report data
        """
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'base_directory': self.base_dir,
            'statistics': {}
        }
        
        try:
            # Collect policy data
            policies_csv = os.path.join(self.base_dir, "all_policies.csv")
            if os.path.exists(policies_csv):
                try:
                    policies_df = pd.read_csv(policies_csv)
                    report['statistics']['total_policies'] = len(policies_df)
                    
                    # Get department statistics
                    departments = policies_df['department'].value_counts().to_dict()
                    report['statistics']['departments'] = departments
                    
                    # Get date range
                    try:
                        dates = pd.to_datetime(policies_df['published_date'], errors='coerce')
                        valid_dates = dates.dropna()
                        if not valid_dates.empty:
                            report['statistics']['earliest_policy'] = valid_dates.min().strftime('%Y-%m-%d')
                            report['statistics']['latest_policy'] = valid_dates.max().strftime('%Y-%m-%d')
                    except Exception as e:
                        self.logger.warning(f"Error processing dates: {e}")
                except Exception as e:
                    self.logger.error(f"Error reading policies CSV: {e}")
            
            # Get category statistics
            category_path = os.path.join(self.base_dir, 'category_structure.json')
            if os.path.exists(category_path):
                try:
                    with open(category_path, 'r', encoding='utf-8') as f:
                        categories = json.load(f)
                    
                    report['statistics']['categories'] = {
                        'total_categories': len(categories),
                        'category_counts': {k: v.get('count', 0) for k, v in categories.items()}
                    }
                except Exception as e:
                    self.logger.error(f"Error reading category structure: {e}")
            
            # Get download statistics
            download_summary = os.path.join(self.base_dir, 'download_summary.json')
            if os.path.exists(download_summary):
                try:
                    with open(download_summary, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    report['statistics']['downloads'] = summary
                except Exception as e:
                    self.logger.error(f"Error reading download summary: {e}")
            
            # Get file statistics
            try:
                total_size = 0
                file_counts = {'pdf': 0, 'doc': 0, 'docx': 0, 'xls': 0, 'xlsx': 0, 'ppt': 0, 'pptx': 0, 'csv': 0, 'other': 0}
                
                for root, dirs, files in os.walk(self.base_dir):
                    # Skip logs and checkpoints directories
                    if 'logs' in root or 'checkpoints' in root:
                        continue
                    
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            # Get file extension
                            ext = os.path.splitext(file)[1].lower()[1:]  # Remove the dot
                            
                            # Count by type
                            if ext in file_counts:
                                file_counts[ext] += 1
                            else:
                                file_counts['other'] += 1
                            
                            # Add to total size
                            if os.path.isfile(file_path):
                                total_size += os.path.getsize(file_path)
                        except Exception as e:
                            self.logger.debug(f"Error processing file {file}: {e}")
                
                # Convert to MB
                total_size_mb = total_size / (1024 * 1024)
                
                report['statistics']['files'] = {
                    'total_size_mb': round(total_size_mb, 2),
                    'file_counts': file_counts,
                    'total_files': sum(file_counts.values())
                }
            except Exception as e:
                self.logger.error(f"Error getting file statistics: {e}")
            
            # Log collection
            try:
                logs_dir = os.path.join(self.base_dir, 'logs')
                log_files = []
                
                if os.path.exists(logs_dir):
                    for log_file in os.listdir(logs_dir):
                        if log_file.endswith('.log'):
                            log_path = os.path.join(logs_dir, log_file)
                            log_files.append({
                                'filename': log_file,
                                'size': os.path.getsize(log_path),
                                'modified': datetime.datetime.fromtimestamp(os.path.getmtime(log_path)).isoformat()
                            })
                
                report['logs'] = {
                    'total_logs': len(log_files),
                    'log_files': log_files
                }
            except Exception as e:
                self.logger.error(f"Error collecting log information: {e}")
            
            # Save report
            report_path = os.path.join(self.base_dir, 'scrape_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Scrape report generated and saved to {report_path}")
            
            return report
        
        except Exception as e:
            self.logger.error(f"Error generating scrape report: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Save partial report
            try:
                partial_report_path = os.path.join(self.base_dir, 'partial_scrape_report.json')
                with open(partial_report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Partial scrape report saved to {partial_report_path}")
            except:
                pass
            
            return report

    def track_department_cooccurrence(self, policies_df=None):
        """
        Analyze policies to identify when multiple departments are listed on the same policy document,
        which may indicate areas of interdepartmental cooperation or potential overlap.
        
        Args:
            policies_df (pd.DataFrame): DataFrame with policy data, or None to load from file
            
        Returns:
            pd.DataFrame: Matrix showing department co-occurrence counts
        """
        self.logger.info("Analyzing department co-occurrences...")
        
        # Load policies from file if not provided
        if policies_df is None:
            csv_path = os.path.join(self.base_dir, "all_policies.csv")
            if not os.path.exists(csv_path):
                self.logger.error(f"Policy data file not found at {csv_path}")
                return pd.DataFrame()
            policies_df = pd.read_csv(csv_path)
        
        # Extract departments (handling cases where multiple departments are listed)
        all_departments = []
        department_pairs = []
        
        for _, policy in policies_df.iterrows():
            dept_str = policy.get('department', '')
            if not dept_str or pd.isna(dept_str):
                continue
                
            # Handle multiple departments listed with separators
            departments = [d.strip() for d in re.split(r'[,;&]', dept_str) if d.strip()]
            all_departments.extend(departments)
            
            # Record all department pairs for co-occurrence
            if len(departments) > 1:
                for i in range(len(departments)):
                    for j in range(i+1, len(departments)):
                        department_pairs.append((departments[i], departments[j]))
        
        # Create co-occurrence matrix
        department_counts = Counter(all_departments)
        top_departments = [dept for dept, _ in department_counts.most_common(30)]
        
        cooccurrence_df = pd.DataFrame(0, index=top_departments, columns=top_departments)
        
        for dept1, dept2 in department_pairs:
            if dept1 in top_departments and dept2 in top_departments:
                cooccurrence_df.loc[dept1, dept2] += 1
                cooccurrence_df.loc[dept2, dept1] += 1
        
        # Save results
        output_path = os.path.join(self.base_dir, "department_cooccurrence.csv")
        try:
            cooccurrence_df.to_csv(output_path)
            self.logger.info(f"Department co-occurrence matrix saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving department co-occurrence matrix: {e}")
        
        return cooccurrence_df

    def analyze_department_categories(self, policies_df=None):
        """
        Analyze which departments are active in which policy categories
        to identify potential interdepartmental overlaps on similar topics.
        
        Args:
            policies_df (pd.DataFrame): DataFrame with policy data, or None to load from file
            
        Returns:
            dict: Nested dictionary with departments, categories, and counts
        """
        self.logger.info("Analyzing department-category relationships...")
        
        # Load policies from file if not provided
        if policies_df is None:
            csv_path = os.path.join(self.base_dir, "all_policies.csv")
            if not os.path.exists(csv_path):
                self.logger.error(f"Policy data file not found at {csv_path}")
                return {}
            policies_df = pd.read_csv(csv_path)
        
        department_categories = {}
        
        for _, policy in policies_df.iterrows():
            department = policy.get('department')
            category = policy.get('category')
            subcategory = policy.get('subcategory')
            
            if not department or pd.isna(department) or not category or pd.isna(category):
                continue
            
            # Clean department name
            department = department.strip()
            
            # Initialize department entry if not exists
            if department not in department_categories:
                department_categories[department] = {'categories': {}, 'total_policies': 0}
            
            # Update category count
            if category not in department_categories[department]['categories']:
                department_categories[department]['categories'][category] = {'count': 0, 'subcategories': {}}
                
            department_categories[department]['categories'][category]['count'] += 1
            department_categories[department]['total_policies'] += 1
            
            # Update subcategory count
            if subcategory and not pd.isna(subcategory):
                if subcategory not in department_categories[department]['categories'][category]['subcategories']:
                    department_categories[department]['categories'][category]['subcategories'][subcategory] = 0
                department_categories[department]['categories'][category]['subcategories'][subcategory] += 1
        
        # Calculate percentage of department's policies in each category
        for dept, data in department_categories.items():
            total = data['total_policies']
            for cat, cat_data in data['categories'].items():
                cat_data['percentage'] = round((cat_data['count'] / total) * 100, 2)
        
        # Save results
        output_path = os.path.join(self.base_dir, "department_categories.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(department_categories, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Department-category analysis saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving department-category analysis: {e}")
        
        return department_categories

    def track_department_activity_over_time(self, policies_df=None):
        """
        Track department policy publication patterns over time to identify
        shifts in departmental focus or coordination.
        
        Args:
            policies_df (pd.DataFrame): DataFrame with policy data, or None to load from file
            
        Returns:
            pd.DataFrame: Time series of department activity
        """
        self.logger.info("Analyzing department activity over time...")
        
        # Load policies from file if not provided
        if policies_df is None:
            csv_path = os.path.join(self.base_dir, "all_policies.csv")
            if not os.path.exists(csv_path):
                self.logger.error(f"Policy data file not found at {csv_path}")
                return pd.DataFrame()
            policies_df = pd.read_csv(csv_path)
        
        # Ensure published_date is in datetime format
        policies_df['date'] = pd.to_datetime(policies_df['published_date'], errors='coerce')
        
        # Filter out rows with invalid dates
        valid_df = policies_df.dropna(subset=['date'])
        
        # Create year and quarter columns
        valid_df['year'] = valid_df['date'].dt.year
        valid_df['quarter'] = valid_df['date'].dt.quarter
        valid_df['year_quarter'] = valid_df['year'].astype(str) + '-Q' + valid_df['quarter'].astype(str)
        
        # Get top departments by volume
        department_counts = valid_df['department'].value_counts().head(15).index.tolist()
        
        # Create time series for each department
        time_periods = sorted(valid_df['year_quarter'].unique())
        dept_activity = pd.DataFrame(index=time_periods)
        
        for dept in department_counts:
            dept_data = valid_df[valid_df['department'] == dept].groupby('year_quarter').size()
            dept_activity[dept] = dept_data
        
        # Save results
        output_path = os.path.join(self.base_dir, "department_activity_time_series.csv")
        try:
            dept_activity.fillna(0).to_csv(output_path)
            self.logger.info(f"Department activity time series saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving department activity time series: {e}")
        
        return dept_activity.fillna(0)

    def identify_keyword_overlaps(self, policies_df=None, keywords_of_interest=None):
        """
        Identify policy overlaps based on key terms appearing in titles and descriptions
        across different departments.
        
        Args:
            policies_df (pd.DataFrame): DataFrame with policy data, or None to load from file
            keywords_of_interest (list): Optional list of keywords to search for
            
        Returns:
            tuple: (keyword_dept_mapping, overlap_stats)
        """
        self.logger.info("Identifying keyword overlaps across departments...")
        
        # Load policies from file if not provided
        if policies_df is None:
            csv_path = os.path.join(self.base_dir, "all_policies.csv")
            if not os.path.exists(csv_path):
                self.logger.error(f"Policy data file not found at {csv_path}")
                return {}, {}
            policies_df = pd.read_csv(csv_path)
        
        if keywords_of_interest is None:
            # Default keywords covering common cross-cutting themes
            keywords_of_interest = [
                'sustainability', 'climate change', 'digital', 'innovation', 
                'infrastructure', 'skills', 'education', 'health', 'economic growth',
                'trade', 'security', 'net zero', 'artificial intelligence', 'AI'
            ]
        
        keyword_dept_mapping = {keyword: {} for keyword in keywords_of_interest}
        
        for _, policy in policies_df.iterrows():
            title = str(policy.get('title', '')).lower()
            desc = str(policy.get('description', '')).lower()
            dept = policy.get('department')
            
            if pd.isna(dept):
                continue
                
            text = title + ' ' + desc
            
            for keyword in keywords_of_interest:
                if keyword.lower() in text:
                    if dept not in keyword_dept_mapping[keyword]:
                        keyword_dept_mapping[keyword][dept] = []
                    
                    keyword_dept_mapping[keyword][dept].append({
                        'title': policy.get('title'),
                        'url': policy.get('url'),
                        'published_date': policy.get('published_date')
                    })
        
        # Calculate overlap statistics
        overlap_stats = {}
        for keyword, dept_dict in keyword_dept_mapping.items():
            depts = list(dept_dict.keys())
            if len(depts) > 1:  # Only include keywords with multiple departments
                overlap_stats[keyword] = {
                    'departments': len(depts),
                    'total_policies': sum(len(policies) for policies in dept_dict.values()),
                    'top_departments': sorted([(d, len(p)) for d, p in dept_dict.items()], 
                                            key=lambda x: x[1], reverse=True)[:5]
                }
        
        # Save results
        output_path = os.path.join(self.base_dir, "keyword_department_overlaps.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'keyword_mapping': {k: {d: len(p) for d, p in v.items()} for k, v in keyword_dept_mapping.items()},
                    'overlap_statistics': overlap_stats
                }, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Keyword overlap analysis saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving keyword overlap analysis: {e}")
        
        return keyword_dept_mapping, overlap_stats

    def generate_department_statistics(self, policies_df=None):
        """
        Generate comprehensive department statistics to enhance the departmental distribution section.
        
        Args:
            policies_df (pd.DataFrame): DataFrame with policy data, or None to load from file
            
        Returns:
            dict: Dictionary with detailed department statistics
        """
        self.logger.info("Generating comprehensive department statistics...")
        
        # Load policies from file if not provided
        if policies_df is None:
            csv_path = os.path.join(self.base_dir, "all_policies.csv")
            if not os.path.exists(csv_path):
                self.logger.error(f"Policy data file not found at {csv_path}")
                return {}
            policies_df = pd.read_csv(csv_path)
        
        # Count policies by department
        dept_counts = policies_df['department'].value_counts()
        
        # Calculate basic statistics
        total_policies = len(policies_df)
        total_departments = len(dept_counts)
        top_departments = dept_counts.head(5)
        top_percentage = (top_departments.sum() / total_policies) * 100
        
        # Identify departments with niche focus
        dept_category = {}
        for _, policy in policies_df.iterrows():
            dept = policy.get('department')
            cat = policy.get('category')
            
            if pd.isna(dept) or pd.isna(cat):
                continue
                
            if dept not in dept_category:
                dept_category[dept] = []
                
            dept_category[dept].append(cat)
        
        # Find departments with narrow focus (>80% in one category)
        niche_departments = {}
        for dept, categories in dept_category.items():
            if len(categories) < 10:  # Skip departments with few policies
                continue
                
            cat_counts = Counter(categories)
            top_cat, top_count = cat_counts.most_common(1)[0]
            percentage = (top_count / len(categories)) * 100
            
            if percentage > 80:  # At least 80% in one category
                niche_departments[dept] = {
                    'primary_category': top_cat,
                    'percentage': percentage,
                    'total_policies': len(categories)
                }
        
        # Final statistics object
        dept_statistics = {
            'total_departments': total_departments,
            'top_5_departments': top_departments.to_dict(),
            'top_5_percentage': top_percentage,
            'niche_departments': niche_departments,
            'departments_by_size': {
                'large': dept_counts[dept_counts >= 100].index.tolist(),
                'medium': dept_counts[(dept_counts >= 30) & (dept_counts < 100)].index.tolist(),
                'small': dept_counts[(dept_counts >= 5) & (dept_counts < 30)].index.tolist(),
                'minimal': dept_counts[dept_counts < 5].index.tolist()
            }
        }
        
        # Save results
        output_path = os.path.join(self.base_dir, "department_statistics.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dept_statistics, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Department statistics saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving department statistics: {e}")
        
        return dept_statistics

    def analyze_departmental_overlaps(self, policies_df=None):
        """
        Main method to run all departmental analysis functions and generate a
        comprehensive report on potential policy overlaps.
        
        Args:
            policies_df (pd.DataFrame): DataFrame with policy data, or None to load from file
            
        Returns:
            dict: Comprehensive analysis results
        """
        self.logger.info("Starting comprehensive departmental overlap analysis...")
        
        # Load policies from file if not provided
        if policies_df is None:
            csv_path = os.path.join(self.base_dir, "all_policies.csv")
            if not os.path.exists(csv_path):
                self.logger.error(f"Policy data file not found at {csv_path}")
                return {}
            policies_df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded {len(policies_df)} policies for analysis")
        
        analysis_results = {}
        
        # Run all analysis functions
        try:
            self.logger.info("Generating basic department statistics...")
            analysis_results['statistics'] = self.generate_department_statistics(policies_df)
            
            self.logger.info("Analyzing department co-occurrences...")
            cooccurrence_df = self.track_department_cooccurrence(policies_df)
            analysis_results['cooccurrence'] = {
                'matrix_shape': cooccurrence_df.shape,
                'total_cooccurrences': cooccurrence_df.sum().sum() // 2,  # Divide by 2 since matrix is symmetric
                'top_pairs': []
            }
            
            # Extract top co-occurring pairs
            for i in range(len(cooccurrence_df.index)):
                for j in range(i+1, len(cooccurrence_df.columns)):
                    dept1 = cooccurrence_df.index[i]
                    dept2 = cooccurrence_df.columns[j]
                    count = cooccurrence_df.iloc[i, j]
                    if count > 0:
                        analysis_results['cooccurrence']['top_pairs'].append((dept1, dept2, int(count)))
            
            # Sort by count and take top 20
            analysis_results['cooccurrence']['top_pairs'].sort(key=lambda x: x[2], reverse=True)
            analysis_results['cooccurrence']['top_pairs'] = analysis_results['cooccurrence']['top_pairs'][:20]
            
            self.logger.info("Analyzing department-category relationships...")
            analysis_results['dept_categories'] = self.analyze_department_categories(policies_df)
            
            self.logger.info("Tracking department activity over time...")
            time_series = self.track_department_activity_over_time(policies_df)
            analysis_results['time_series'] = {
                'shape': time_series.shape,
                'periods': len(time_series.index),
                'departments': time_series.columns.tolist()
            }
            
            self.logger.info("Identifying keyword overlaps...")
            _, overlap_stats = self.identify_keyword_overlaps(policies_df)
            analysis_results['keyword_overlaps'] = overlap_stats
            
            # Identify key overlap findings
            self.logger.info("Synthesizing key overlap findings...")
            analysis_results['key_findings'] = self._synthesize_key_findings(analysis_results)
            
        except Exception as e:
            self.logger.error(f"Error during departmental overlap analysis: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Save comprehensive report
        report_path = os.path.join(self.base_dir, "departmental_overlap_analysis.json")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Comprehensive departmental overlap analysis saved to {report_path}")
        except Exception as e:
            self.logger.error(f"Error saving departmental overlap analysis: {e}")
        
        return analysis_results

    def _synthesize_key_findings(self, analysis_results):
        """
        Synthesize key findings from the various analyses to highlight
        the most significant overlaps and patterns.
        
        Args:
            analysis_results (dict): Results from various analysis functions
            
        Returns:
            dict: Key findings and insights
        """
        findings = {
            'top_5_concentration': None,
            'most_collaborative_departments': [],
            'most_isolated_departments': [],
            'highest_category_overlaps': [],
            'trending_keywords_across_departments': [],
            'departments_with_changing_focus': []
        }
        
        # Extract top 5 concentration
        if 'statistics' in analysis_results and 'top_5_percentage' in analysis_results['statistics']:
            findings['top_5_concentration'] = round(analysis_results['statistics']['top_5_percentage'], 2)
        
        # Identify most collaborative departments (appear most often in co-occurrences)
        if 'cooccurrence' in analysis_results and 'top_pairs' in analysis_results['cooccurrence']:
            dept_counts = Counter()
            for dept1, dept2, count in analysis_results['cooccurrence']['top_pairs']:
                dept_counts[dept1] += count
                dept_counts[dept2] += count
            
            findings['most_collaborative_departments'] = dept_counts.most_common(5)
        
        # Identify highest category overlaps
        if 'dept_categories' in analysis_results:
            category_depts = {}
            for dept, data in analysis_results['dept_categories'].items():
                if 'categories' not in data:
                    continue
                    
                for cat, cat_data in data['categories'].items():
                    if cat not in category_depts:
                        category_depts[cat] = []
                    
                    if cat_data['percentage'] >= 15:  # Department has significant activity in this category
                        category_depts[cat].append((dept, cat_data['count'], cat_data['percentage']))
            
            # Find categories with most department overlap
            category_overlap = [(cat, depts) for cat, depts in category_depts.items() if len(depts) >= 3]
            category_overlap.sort(key=lambda x: len(x[1]), reverse=True)
            
            findings['highest_category_overlaps'] = [
                {
                    'category': cat,
                    'department_count': len(depts),
                    'top_departments': sorted(depts, key=lambda x: x[1], reverse=True)[:5]
                }
                for cat, depts in category_overlap[:10]
            ]
        
        # Extract trending keywords
        if 'keyword_overlaps' in analysis_results:
            sorted_keywords = sorted(
                analysis_results['keyword_overlaps'].items(),
                key=lambda x: x[1]['departments'] * x[1]['total_policies'],
                reverse=True
            )
            
            findings['trending_keywords_across_departments'] = [
                {
                    'keyword': k,
                    'departments': v['departments'],
                    'total_policies': v['total_policies'],
                    'top_departments': v['top_departments'][:3]
                }
                for k, v in sorted_keywords[:10]
            ]
        
        return findings

    def generate_department_summary_for_thesis(self):
        """
        Generate a formatted summary of departmental distribution and overlaps
        that can be directly used in the thesis document.
        
        Returns:
            str: Formatted text for thesis inclusion
        """
        self.logger.info("Generating department summary for thesis...")
        
        # Try to load the comprehensive analysis results
        analysis_path = os.path.join(self.base_dir, "departmental_overlap_analysis.json")
        if not os.path.exists(analysis_path):
            self.logger.warning(f"Analysis file not found at {analysis_path}, running analysis first...")
            self.analyze_departmental_overlaps()
            if not os.path.exists(analysis_path):
                self.logger.error("Failed to generate department analysis")
                return "Analysis data not available."
        
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading analysis data: {e}")
            return "Error loading analysis data."
        
        # Format the summary text
        summary = []
        
        # Section header
        summary.append("### 3.2.2 Departmental Distribution\n")
        
        # Basic statistics
        if 'statistics' in analysis and 'top_5_percentage' in analysis['statistics']:
            stats = analysis['statistics']
            summary.append(f"The dataset encompasses policies from {stats['total_departments']} different UK government departments and agencies. The distribution of policies across departments is notably uneven, with the top 5 departments accounting for {stats['top_5_percentage']:.1f}% of all policy papers. These dominant departments include {', '.join(list(stats['top_5_departments'].keys())[:3])} and others, reflecting their central role in policy development across multiple domains.\n")
        
        # Department size categories
        if 'statistics' in analysis and 'departments_by_size' in analysis['statistics']:
            size_stats = analysis['statistics']['departments_by_size']
            summary.append(f"The departments can be categorized by their policy publication volume: {len(size_stats.get('large', []))} large departments (100+ policies each), {len(size_stats.get('medium', []))} medium departments (30-99 policies), {len(size_stats.get('small', []))} smaller departments (5-29 policies), and {len(size_stats.get('minimal', []))} departments with minimal policy output (fewer than 5 policies).\n")
        
        # Niche departments
        if 'statistics' in analysis and 'niche_departments' in analysis['statistics']:
            niche_depts = analysis['statistics']['niche_departments']
            if niche_depts:
                niche_examples = list(niche_depts.items())[:3]
                niche_text = ", ".join([f"{dept} (focusing {data['percentage']:.1f}% on {data['primary_category']})" for dept, data in niche_examples])
                summary.append(f"Several departments exhibit highly specialized focus areas, with over 80% of their publications concentrated in a single policy domain. Notable examples include {niche_text}. This specialization contrasts with larger departments that publish across a broader spectrum of policy areas.\n")
        
        # Interdepartmental overlaps
        if 'highest_category_overlaps' in analysis.get('key_findings', {}):
            overlaps = analysis['key_findings']['highest_category_overlaps']
            if overlaps:
                top_overlap = overlaps[0]
                dept_names = [d[0] for d in top_overlap['top_departments'][:3]]
                summary.append(f"The analysis reveals significant interdepartmental overlaps in several policy domains. The most prominent overlap occurs in {top_overlap['category']}, where {top_overlap['department_count']} different departments actively publish policies, including {', '.join(dept_names)}. This finding suggests potential areas for enhanced coordination or consolidation of policy efforts.\n")
        
        # Keyword overlaps
        if 'trending_keywords_across_departments' in analysis.get('key_findings', {}):
            keyword_trends = analysis['key_findings']['trending_keywords_across_departments']
            if keyword_trends:
                cross_cutting_keywords = [item['keyword'] for item in keyword_trends[:5]]
                summary.append(f"Cross-cutting policy themes identified through keyword analysis include {', '.join(cross_cutting_keywords)}. These themes appear in publications from multiple departments, suggesting both collaborative opportunities and potential duplication risks in policy development.\n")
        
        # Collaborative departments
        if 'most_collaborative_departments' in analysis.get('key_findings', {}):
            collab_depts = analysis['key_findings']['most_collaborative_departments']
            if collab_depts:
                collab_examples = [name for name, _ in collab_depts[:3]]
                summary.append(f"The most collaborative departments, based on co-authorship patterns, are {', '.join(collab_examples)}. These departments frequently publish joint policy papers, demonstrating existing interdepartmental coordination mechanisms.\n")
        
        # Combine all sections
        full_summary = "\n".join(summary)
        
        # Save the summary
        summary_path = os.path.join(self.base_dir, "departmental_summary_for_thesis.txt")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(full_summary)
            self.logger.info(f"Department summary for thesis saved to {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving department summary: {e}")
        
        return full_summary

def main():
    """
    Main function to run the scraper with comprehensive failsafe mechanisms
    """
    # Setup exception handling to log all uncaught exceptions
    import sys
    
    # Create base directories
    base_dir = "/Volumes/Research/policy_data"
    os.makedirs(base_dir, exist_ok=True)
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup a global exception logger
    global_log_path = os.path.join(logs_dir, f'global_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    def exception_handler(exc_type, exc_value, exc_traceback):
        # Log the exception
        with open(global_log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n\n===== UNCAUGHT EXCEPTION at {timestamp} =====\n")
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write("\n===== END OF EXCEPTION =====\n")
        
        # Also print to stderr
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    # Set the exception handler
    sys.excepthook = exception_handler
    
    # Initialize the scraper
    scraper = GovUKPolicyScraper(base_dir=base_dir)
    
    # Track overall success/failure
    results = {
        "start_time": datetime.datetime.now().isoformat(),
        "policies_found": 0,
        "attachments_downloaded": 0,
        "failed_downloads": 0,
        "categories_found": 0,
        "visualizations_created": 0,
        "errors": [],
        "warnings": [],
        "completion_status": "in_progress"
    }
    
    # Function to save progress results
    def save_results():
        results["last_updated"] = datetime.datetime.now().isoformat()
        results_path = os.path.join(base_dir, 'scraping_results.json')
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            scraper.logger.info(f"Saved scraping results to {results_path}")
        except Exception as e:
            scraper.logger.error(f"Error saving scraping results: {e}")
    
    try:
        # Step 1: Search and extract policies
        scraper.logger.info("STEP 1: Starting policy search and extraction")
        try:
            # Check for existing policy data
            csv_path = os.path.join(base_dir, "all_policies.csv")
            if os.path.exists(csv_path):
                scraper.logger.info(f"Found existing policy data at {csv_path}")
                try:
                    policies = pd.read_csv(csv_path)
                    results["policies_found"] = len(policies)
                    scraper.logger.info(f"Loaded {len(policies)} existing policies")
                except Exception as e:
                    scraper.logger.error(f"Error loading existing policy data: {e}")
                    # Continue with search anyway
                    policies = scraper.search_policies(
                        search_term=None,  # Set a search term or None for all policies
                        page_limit=1,      # Number of pages to scrape
                        
                        exclude_types=["consultation", "open consultation", "closed consultation"]
                    )
                    results["policies_found"] = len(policies)
            else:
                # No existing data, perform search
                policies = scraper.search_policies(
                    search_term=None,  # Set a search term or None for all policies
                    page_limit=1000,     # Number of pages to scrape THIS IS THE ONE CHANGE THIS VALUE!!! MARKER
                    exclude_types=["consultation", "open consultation", "closed consultation"]
                )
                results["policies_found"] = len(policies)
        except Exception as e:
            scraper.logger.error(f"Error during policy search: {e}")
            scraper.logger.debug(f"Traceback: {traceback.format_exc()}")
            policies = pd.DataFrame()  # Create empty DataFrame to continue
            results["errors"].append({"stage": "search_policies", "error": str(e), "traceback": traceback.format_exc()})
        
        # Save progress
        save_results()
        
        # Step 2: Categorize the policies
        if not policies.empty:
            scraper.logger.info("STEP 2: Starting policy categorization")
            try:
                categories = scraper.categorize_policies(policies)
                results["categories_found"] = len(categories)
            except Exception as e:
                scraper.logger.error(f"Error during policy categorization: {e}")
                scraper.logger.debug(f"Traceback: {traceback.format_exc()}")
                categories = {}  # Create empty dict to continue
                results["errors"].append({"stage": "categorize_policies", "error": str(e), "traceback": traceback.format_exc()})
        else:
            categories = {}
            results["warnings"].append("Skipped categorization due to empty policy list")
        
        # Save progress
        save_results()
        
        # Step 3: Download and organize attachments
        if not policies.empty:
            scraper.logger.info("STEP 3: Starting attachment downloads")
            try:
                download_count, failed_downloads, success_info = scraper.download_policy_attachments(policies)
                results["attachments_downloaded"] = download_count
                results["failed_downloads"] = len(failed_downloads)
                results["download_success_info"] = len(success_info)
            except Exception as e:
                scraper.logger.error(f"Error during attachment download: {e}")
                scraper.logger.debug(f"Traceback: {traceback.format_exc()}")
                results["errors"].append({"stage": "download_policy_attachments", "error": str(e), "traceback": traceback.format_exc()})
        else:
            results["warnings"].append("Skipped downloading attachments due to empty policy list")
        
        # Save progress
        save_results()
        
        # Step 4: Create visualizations
        scraper.logger.info("STEP 4: Generating dendrogram visualizations")
        try:
            # Import here to avoid early failure if module is missing
            from policy_dendrogram import PolicyDendrogramVisualizer
            visualizer = PolicyDendrogramVisualizer(base_dir=scraper.base_dir)
            viz_paths = visualizer.create_all_visualizations()
            results["visualizations_created"] = len(viz_paths)
            scraper.logger.info(f"Created {len(viz_paths)} visualizations")
        except ImportError as e:
            scraper.logger.warning(f"Visualization module not available: {e}")
            results["warnings"].append({"stage": "create_visualizations", "warning": "Visualization module not available"})
        except Exception as e:
            scraper.logger.error(f"Error generating visualizations: {e}")
            scraper.logger.debug(f"Traceback: {traceback.format_exc()}")
            results["errors"].append({"stage": "create_visualizations", "error": str(e), "traceback": traceback.format_exc()})
        
        # Step 4a: Advanced departmental analysis
        scraper.logger.info("STEP 4a: Performing advanced departmental analysis")
        try:
            if not policies.empty:
                analysis_results = scraper.analyze_departmental_overlaps(policies)
                results["departmental_analysis"] = True
                results["overlap_findings"] = len(analysis_results.get('key_findings', {}).get('highest_category_overlaps', []))
                scraper.logger.info(f"Completed departmental analysis with {results['overlap_findings']} overlap findings")
            else:
                results["warnings"].append("Skipped departmental analysis due to empty policy list")
        except Exception as e:
            scraper.logger.error(f"Error during departmental analysis: {e}")
            scraper.logger.debug(f"Traceback: {traceback.format_exc()}")
            results["errors"].append({"stage": "departmental_analysis", "error": str(e), "traceback": traceback.format_exc()})

        # Save progress
        save_results()

        # Step 5: Generate comprehensive report
        scraper.logger.info("STEP 5: Generating comprehensive scrape report")
        try:
            report = scraper.generate_scrape_report()
            results["report_generated"] = True
        except Exception as e:
            scraper.logger.error(f"Error generating report: {e}")
            results["errors"].append({"stage": "generate_report", "error": str(e)})
        
        # Completed successfully
        results["completion_status"] = "completed"
        scraper.logger.info("Scraping process completed successfully")
        
    except KeyboardInterrupt:
        scraper.logger.warning("\nProcess interrupted by user.")
        results["completion_status"] = "interrupted"
        results["errors"].append({"stage": "main", "error": "Process interrupted by user"})
    except Exception as e:
        scraper.logger.error(f"\nUnexpected error in main process: {e}")
        scraper.logger.debug(f"Traceback: {traceback.format_exc()}")
        results["completion_status"] = "failed"
        results["errors"].append({"stage": "main", "error": str(e), "traceback": traceback.format_exc()})
    finally:
        # Final summary regardless of success or failure
        scraper.logger.info("\nScraping Summary:")
        scraper.logger.info(f"Total policies found: {results['policies_found']}")
        scraper.logger.info(f"Total categories: {results['categories_found']}")
        scraper.logger.info(f"Total attachments downloaded: {results['attachments_downloaded']}")
        scraper.logger.info(f"Failed downloads: {results['failed_downloads']}")
        scraper.logger.info(f"Total errors: {len(results['errors'])}")
        scraper.logger.info(f"Total warnings: {len(results['warnings'])}")
        scraper.logger.info(f"Completion status: {results['completion_status']}")
        
        # Add end time
        results["end_time"] = datetime.datetime.now().isoformat()
        
        # Calculate duration
        start_time = datetime.datetime.fromisoformat(results["start_time"])
        end_time = datetime.datetime.fromisoformat(results["end_time"])
        duration = (end_time - start_time).total_seconds()
        results["duration_seconds"] = duration
        results["duration_formatted"] = f"{duration // 3600:.0f}h {(duration % 3600) // 60:.0f}m {duration % 60:.0f}s"
        
        # Final save
        save_results()
        
        # Create a one-line success/failure flag file for automated monitoring
        status_path = os.path.join(base_dir, f'status_{results["completion_status"]}.txt')
        try:
            with open(status_path, 'w', encoding='utf-8') as f:
                f.write(f"Scraper {results['completion_status']} at {results['end_time']} with {len(results['errors'])} errors")
        except:
            pass
    

        # Generate thesis-ready departmental summary
        try:
            thesis_summary = scraper.generate_department_summary_for_thesis()
            results["thesis_summary_generated"] = True
            scraper.logger.info("Generated department summary for thesis")
        except Exception as e:
            scraper.logger.error(f"Error generating thesis summary: {e}")
            results["errors"].append({"stage": "thesis_summary", "error": str(e)})

    return results

if __name__ == "__main__":
    main()