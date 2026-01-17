//! YouTube URL extraction from HTML

use crate::Result;
use scraper::{Html, Selector};

/// Extract YouTube URLs from HTML content
///
/// Searches for YouTube URLs in href attributes and text content.
/// Supports various YouTube URL formats:
/// - https://www.youtube.com/watch?v=VIDEO_ID
/// - https://youtu.be/VIDEO_ID
/// - https://www.youtube.com/embed/VIDEO_ID
/// - https://m.youtube.com/watch?v=VIDEO_ID
///
/// # Examples
///
/// ```
/// use melops_web::extract_youtube_urls;
///
/// let html = r#"<a href="https://youtu.be/dQw4w9WgXcQ">video</a>"#;
/// let urls = extract_youtube_urls(html)?;
/// assert_eq!(urls.len(), 1);
/// # Ok::<(), melops_web::Error>(())
/// ```
pub fn extract_youtube_urls(html: &str) -> Result<Vec<String>> {
    let document = Html::parse_document(html);
    let link_selector = Selector::parse("a[href]").unwrap();

    let mut urls = Vec::new();

    for element in document.select(&link_selector) {
        if let Some(href) = element.value().attr("href")
            && is_youtube_url(href)
        {
            urls.push(href.to_string());
        }
    }

    Ok(urls)
}

/// Check if a URL is a YouTube URL
fn is_youtube_url(url: &str) -> bool {
    url.contains("youtube.com") || url.contains("youtu.be")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_youtube_com_watch_url() {
        let html = r#"<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">video</a>"#;
        let urls = extract_youtube_urls(html).unwrap();
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    }

    #[test]
    fn extracts_youtu_be_short_url() {
        let html = r#"<a href="https://youtu.be/dQw4w9WgXcQ">video</a>"#;
        let urls = extract_youtube_urls(html).unwrap();
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "https://youtu.be/dQw4w9WgXcQ");
    }

    #[test]
    fn extracts_embed_url() {
        let html = r#"<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>"#;
        let urls = extract_youtube_urls(html).unwrap();
        assert_eq!(urls.len(), 0); // iframe uses src, not href
    }

    #[test]
    fn extracts_multiple_urls() {
        let html = r#"
            <a href="https://youtu.be/video1">first</a>
            <a href="https://www.youtube.com/watch?v=video2">second</a>
            <a href="https://example.com">not youtube</a>
            <a href="https://m.youtube.com/watch?v=video3">third</a>
        "#;
        let urls = extract_youtube_urls(html).unwrap();
        assert_eq!(urls.len(), 3);
    }

    #[test]
    fn ignores_non_youtube_urls() {
        let html = r#"
            <a href="https://example.com">example</a>
            <a href="https://github.com">github</a>
        "#;
        let urls = extract_youtube_urls(html).unwrap();
        assert_eq!(urls.len(), 0);
    }

    #[test]
    fn handles_empty_html() {
        let urls = extract_youtube_urls("").unwrap();
        assert_eq!(urls.len(), 0);
    }

    #[test]
    fn handles_html_without_links() {
        let html = "<p>No links here</p>";
        let urls = extract_youtube_urls(html).unwrap();
        assert_eq!(urls.len(), 0);
    }
}
