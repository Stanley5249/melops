# melops-web

Minimal web scraping utilities for extracting YouTube URLs from HTML pages.

## Features

- **Fetch**: Download HTML content from URLs using `reqwest`
- **Extract**: Parse HTML and extract YouTube URLs using `scraper`
- **Resolve**: Resolve URLs to canonical form via HTTP HEAD requests

## Design Philosophy

This is a **minimal library** that provides basic building blocks. All orchestration, caching, and validation logic belongs in the CLI layer (`melops`).

## Usage

```rust
use melops_web::{fetch_page, extract_youtube_urls, resolve_url};

// Fetch page
let html = fetch_page("https://example.com/course")?;

// Extract YouTube URLs
let urls = extract_youtube_urls(&html)?;

// Resolve each URL to canonical form
for url in urls {
    let resolved = resolve_url(&url)?;
    println!("{}", resolved);
}
```

## Error Handling

Uses `thiserror` with transparent error propagation:

- `Http(reqwest::Error)` - Network and HTTP errors
- `Io(std::io::Error)` - Filesystem errors

## URL Resolution

Uses HTTP HEAD requests to resolve:

- Short URLs (`youtu.be`) → canonical URLs
- Mobile URLs (`m.youtube.com`) → desktop URLs
- Redirects → final destination

Falls back to original URL if HTTP request fails.
