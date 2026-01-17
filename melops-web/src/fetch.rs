//! HTTP page fetching

use crate::Result;

/// Fetch HTML content from a URL
///
/// Uses reqwest blocking client to fetch the page content.
/// Follows redirects automatically.
///
/// # Examples
///
/// ```no_run
/// use melops_web::fetch_page;
///
/// let html = fetch_page("https://example.com")?;
/// # Ok::<(), melops_web::Error>(())
/// ```
pub fn fetch_page(url: &str) -> Result<String> {
    let response = reqwest::blocking::get(url)?;
    let html = response.text()?;
    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "network I/O"]
    fn fetches_example_page() {
        let html = fetch_page("https://example.com").unwrap();
        assert!(html.contains("Example Domain"));
    }
}
