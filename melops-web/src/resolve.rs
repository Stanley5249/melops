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
/// # use melops_web::resolve_url;
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let resolved = resolve_url("https://youtu.be/dQw4w9WgXcQ").await.unwrap();
/// assert!(resolved.contains("youtube.com/watch"));
/// # });
/// ```
pub async fn resolve_url(url: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    match client.head(url).send().await {
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

    #[tokio::test]
    #[ignore = "network I/O"]
    async fn resolves_short_url() {
        let resolved = resolve_url("https://youtu.be/dQw4w9WgXcQ").await.unwrap();
        assert!(resolved.contains("youtube.com"));
    }

    #[tokio::test]
    #[ignore = "network I/O"]
    async fn resolves_canonical_url() {
        let url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
        let resolved = resolve_url(url).await.unwrap();
        assert_eq!(resolved, url);
    }

    #[tokio::test]
    #[ignore = "network I/O"]
    async fn handles_invalid_url() {
        let url = "https://invalid.example.com/nonexistent";
        let resolved = resolve_url(url).await.unwrap();
        // Falls back to original on error
        assert_eq!(resolved, url);
    }
}
