//! URL resolution to canonical form

use crate::Result;

/// Resolve URL to canonical form using HTTP HEAD request
///
/// Follows redirects to get the final URL. This is useful for:
/// - Short URLs (youtu.be) → canonical URLs
/// - Mobile URLs (m.youtube.com) → desktop URLs
/// - Redirects → final destination
///
/// Falls back to original URL if HTTP request fails.
///
/// # Examples
///
/// ```no_run
/// use melops_web::resolve_url;
///
/// let resolved = resolve_url("https://youtu.be/dQw4w9WgXcQ")?;
/// assert!(resolved.contains("youtube.com/watch"));
/// # Ok::<(), melops_web::Error>(())
/// ```
pub fn resolve_url(url: &str) -> Result<String> {
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    match client.head(url).send() {
        Ok(response) => Ok(response.url().to_string()),
        Err(_) => {
            // Fallback to original URL if HTTP fails
            Ok(url.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "network I/O"]
    fn resolves_short_url() {
        let resolved = resolve_url("https://youtu.be/dQw4w9WgXcQ").unwrap();
        assert!(resolved.contains("youtube.com"));
    }

    #[test]
    #[ignore = "network I/O"]
    fn resolves_canonical_url() {
        let url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
        let resolved = resolve_url(url).unwrap();
        assert_eq!(resolved, url);
    }

    #[test]
    #[ignore = "network I/O"]
    fn handles_invalid_url() {
        let url = "https://invalid.example.com/nonexistent";
        let resolved = resolve_url(url).unwrap();
        // Falls back to original on error
        assert_eq!(resolved, url);
    }
}
