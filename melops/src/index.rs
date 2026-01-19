//! Artifact index management with generic resource caching.
//!
//! The index tracks three types of artifacts:
//! - Web pages → YouTube URLs (page scraping results)
//! - YouTube URLs → audio files (downloaded audio paths)
//! - Audio files → SRT files (generated caption paths)

use crate::cache::{CacheDir, INDEX_FILENAME};
use crate::cli::IndexArgs;
use clap::ValueEnum;
use eyre::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::future::Future;
use std::hash::Hash;
use std::path::PathBuf;

/// Cache operation strategy for resource access.
///
/// Controls how the index handles cached artifacts:
/// - `Get`: Read-only, fails if not cached (use with `--cache-*=get`)
/// - `GetOrInsert`: Default behavior, computes if missing
/// - `Replace`: Forces recomputation, ignores cache (use with `--cache-*=replace`)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum CacheStrategy {
    /// Use existing cached value only (error if missing)
    Get,
    /// Use cache or compute if missing (default)
    #[default]
    GetOrInsert,
    /// Always recompute and update cache
    Replace,
}

/// Generic resource cache with lazy factory pattern
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceCache<K, V>
where
    K: Hash + Eq,
{
    inner: HashMap<K, V>,
}

impl<K, V> ResourceCache<K, V>
where
    K: Hash + Eq,
{
    /// Create empty cache
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Access resource with cache operation and async factory
    ///
    /// - `CacheStrategy::Get`: Return cached value only (error if missing)
    /// - `CacheStrategy::GetOrInsert`: Return cached value or compute with factory
    /// - `CacheStrategy::Replace`: Always compute with factory and update cache
    ///
    /// The factory does all the work (compute + persist to disk if needed)
    /// and returns the data to be cached.
    ///
    /// Uses Entry API to avoid hashing the key multiple times.
    pub async fn access<F, Fut>(&mut self, key: K, op: CacheStrategy, factory: F) -> Result<&V>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<V>>,
    {
        match op {
            CacheStrategy::Get => match self.inner.get(&key) {
                Some(value) => Ok(value),
                None => Err(eyre::eyre!("resource not found in cache")),
            },
            CacheStrategy::GetOrInsert => match self.inner.entry(key) {
                Entry::Occupied(entry) => Ok(entry.into_mut()),
                Entry::Vacant(entry) => {
                    let value = factory().await?;
                    Ok(entry.insert(value))
                }
            },
            CacheStrategy::Replace => {
                let value = factory().await?;
                Ok(match self.inner.entry(key) {
                    Entry::Occupied(mut entry) => {
                        entry.insert(value);
                        entry.into_mut()
                    }
                    Entry::Vacant(entry) => entry.insert(value),
                })
            }
        }
    }

    /// Get cached value without factory (read-only access)
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.get(key)
    }

    /// Insert or update value
    pub fn insert(&mut self, key: K, value: V) {
        self.inner.insert(key, value);
    }
}

/// Three-level artifact index data
///
/// - pages: web page URL → resolved YouTube URLs
/// - audio: YouTube URL → downloaded audio path
/// - srt: audio path (canonical) → generated SRT path
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct IndexData {
    pub pages: ResourceCache<String, Vec<String>>,
    pub audio: ResourceCache<String, PathBuf>,
    pub srt: ResourceCache<String, PathBuf>,
}

/// Artifact index with persistent storage and cache operations
pub struct ArtifactCache {
    data: IndexData,
    path: PathBuf,
    cache_pages: CacheStrategy,
    cache_audio: CacheStrategy,
    cache_srt: CacheStrategy,
}

impl ArtifactCache {
    /// Access web page URLs cache with factory
    pub async fn ensure_pages<F, Fut>(
        &mut self,
        page_url: String,
        factory: F,
    ) -> Result<&Vec<String>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Vec<String>>>,
    {
        self.data
            .pages
            .access(page_url, self.cache_pages, factory)
            .await
    }

    /// Access audio files cache with factory
    pub async fn ensure_audio<F, Fut>(&mut self, url: String, factory: F) -> Result<&PathBuf>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<PathBuf>>,
    {
        self.data.audio.access(url, self.cache_audio, factory).await
    }

    /// Access SRT files cache with factory
    pub async fn ensure_srt<F, Fut>(&mut self, cache_key: String, factory: F) -> Result<&PathBuf>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<PathBuf>>,
    {
        self.data
            .srt
            .access(cache_key, self.cache_srt, factory)
            .await
    }
}

impl std::ops::Deref for ArtifactCache {
    type Target = IndexData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for ArtifactCache {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl ArtifactCache {
    /// Load index from cache directory and index arguments
    pub fn load(cache_dir: CacheDir, index_args: IndexArgs) -> Result<Self> {
        let path = cache_dir.join(INDEX_FILENAME);

        let data = if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            serde_json::from_str(&content)?
        } else {
            IndexData::default()
        };

        Ok(Self {
            data,
            path,
            cache_pages: index_args.cache_pages,
            cache_audio: index_args.cache_audio,
            cache_srt: index_args.cache_srt,
        })
    }

    /// Save index to disk
    fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(&self.path, content)?;
        Ok(())
    }
}

impl Drop for ArtifactCache {
    fn drop(&mut self) {
        if let Err(e) = self.save() {
            tracing::error!(error = %e, "failed to save index");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resource_cache_get_returns_error_if_missing() {
        let mut cache = ResourceCache::<&str, i32>::new();

        let result = cache
            .access("key", CacheStrategy::Get, async || {
                panic!("factory should not be called for Get")
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn resource_cache_get_returns_cached() {
        let mut cache = ResourceCache::new();
        cache.insert("key", 42);

        let result = cache
            .access("key", CacheStrategy::Get, async || {
                panic!("factory should not be called for cached Get")
            })
            .await
            .unwrap();
        assert_eq!(result, &42);
    }

    #[tokio::test]
    async fn resource_cache_get_or_insert_caches_result() {
        let mut cache = ResourceCache::new();

        let result1 = cache
            .access("key", CacheStrategy::GetOrInsert, async || {
                Ok::<_, eyre::Report>(42)
            })
            .await
            .unwrap();
        assert_eq!(result1, &42);

        // Second access should return cached value
        let result2 = cache
            .access("key", CacheStrategy::GetOrInsert, async || {
                panic!("factory should not be called for cached value")
            })
            .await
            .unwrap();
        assert_eq!(result2, &42);
    }

    #[tokio::test]
    async fn resource_cache_replace_recomputes() {
        let mut cache = ResourceCache::new();
        cache.insert("key", 42);

        let result = cache
            .access("key", CacheStrategy::Replace, async || {
                Ok::<_, eyre::Report>(99)
            })
            .await
            .unwrap();
        assert_eq!(result, &99);
    }

    #[test]
    fn resource_cache_serialization() {
        let mut cache = ResourceCache::new();
        cache.insert("key1".to_string(), "value1".to_string());
        cache.insert("key2".to_string(), "value2".to_string());

        let json = serde_json::to_string(&cache).unwrap();
        let loaded: ResourceCache<String, String> = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.get("key1"), Some(&"value1".to_string()));
        assert_eq!(loaded.get("key2"), Some(&"value2".to_string()));
    }
}
