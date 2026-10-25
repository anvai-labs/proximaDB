//! `FileSystem` decorators forward most methods unchanged to the filesystem
//! they wrap; [`forward_filesystem_impl!`](crate::forward_filesystem_impl)
//! writes those forwards once.
//!
//! Each decorator (`CountingFileSystem`, `FaultInjectingFileSystem`, the
//! latency test wrapper, …) used to re-type the same twenty pass-through
//! methods, and every new decorator copied them again. The macro emits the
//! whole `#[async_trait] impl FileSystem` — the listed forwards plus the
//! methods the decorator actually changes — in a single expansion, so
//! `async_trait` only ever sees concrete `fn` items (an inner macro call left
//! inside the impl would reach `async_trait` unexpanded).

/// Implement `FileSystem` for a decorator.
///
/// ```ignore
/// forward_filesystem_impl! {
///     impl FileSystem for MyWrapper => inner;
///     forward [as_any, filesystem_type, write, delete, sync];
///     {
///         async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
///             /* the behavior the wrapper adds */
///         }
///     }
/// }
/// ```
///
/// `inner` names the wrapped field (anything that derefs to a `FileSystem`).
/// Each method in `forward [..]` becomes `self.inner.<method>(..)` with the
/// trait's exact signature; the block holds every other method the decorator
/// implements. A method neither forwarded nor written keeps the trait default.
#[macro_export]
macro_rules! forward_filesystem_impl {
    (
        impl FileSystem for $ty:ty => $inner:ident;
        forward [$($method:ident),* $(,)?];
        { $($custom:tt)* }
    ) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($custom)*], [$($method),*]);
    };

    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], []) => {
        #[$crate::__async_trait::async_trait]
        impl $crate::FileSystem for $ty {
            $($acc)*
        }
    };

    // ---- synchronous methods --------------------------------------------
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [as_any $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            fn as_any(&self) -> &dyn std::any::Any {
                self.$inner.as_any()
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [filesystem_type $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            fn filesystem_type(&self) -> &'static str {
                self.$inner.filesystem_type()
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [supports_mmap $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            fn supports_mmap(&self) -> bool {
                self.$inner.supports_mmap()
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [range_coalesce_policy $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            fn range_coalesce_policy(&self) -> Option<$crate::RangeCoalescePolicy> {
                self.$inner.range_coalesce_policy()
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [supports_bounded_local_file_write $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            fn supports_bounded_local_file_write(&self) -> bool {
                self.$inner.supports_bounded_local_file_write()
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [supports_append $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            fn supports_append(&self) -> bool {
                self.$inner.supports_append()
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [supports_conditional_replace $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            fn supports_conditional_replace(&self) -> bool {
                self.$inner.supports_conditional_replace()
            }
        ], [$($rest),*]);
    };

    // ---- reads --------------------------------------------------------------
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [read_ranges $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn read_ranges(
                &self,
                path: &str,
                ranges: Vec<std::ops::Range<u64>>,
            ) -> $crate::FsResult<Vec<Vec<u8>>> {
                self.$inner.read_ranges(path, ranges).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [get_mmap $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn get_mmap(&self, path: &str) -> $crate::FsResult<Option<$crate::__memmap2::Mmap>> {
                self.$inner.get_mmap(path).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [exists $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn exists(&self, path: &str) -> $crate::FsResult<bool> {
                self.$inner.exists(path).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [metadata $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn metadata(&self, path: &str) -> $crate::FsResult<$crate::FsFileMetadata> {
                self.$inner.metadata(path).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [list $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn list(&self, path: &str) -> $crate::FsResult<Vec<$crate::DirEntry>> {
                self.$inner.list(path).await
            }
        ], [$($rest),*]);
    };

    // ---- writes, handles and namespace operations -------------------------
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [write $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn write(
                &self,
                path: &str,
                data: &[u8],
                options: Option<$crate::FileOptions>,
            ) -> $crate::FsResult<()> {
                self.$inner.write(path, data, options).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [write_local_file $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn write_local_file(
                &self,
                path: &str,
                local_path: &std::path::Path,
                options: Option<$crate::FileOptions>,
            ) -> $crate::FsResult<u64> {
                self.$inner.write_local_file(path, local_path, options).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [write_if_absent $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn write_if_absent(
                &self,
                path: &str,
                data: &[u8],
                options: Option<$crate::FileOptions>,
            ) -> $crate::FsResult<()> {
                self.$inner.write_if_absent(path, data, options).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [append $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn append(&self, path: &str, data: &[u8]) -> $crate::FsResult<()> {
                self.$inner.append(path, data).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [compare_exchange $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn compare_exchange(
                &self,
                path: &str,
                expected: Option<&[u8]>,
                replacement: &[u8],
                guards: &[(&str, Option<&[u8]>)],
            ) -> $crate::FsResult<bool> {
                self.$inner
                    .compare_exchange(path, expected, replacement, guards)
                    .await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [delete $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn delete(&self, path: &str) -> $crate::FsResult<()> {
                self.$inner.delete(path).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [create_dir $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn create_dir(&self, path: &str) -> $crate::FsResult<()> {
                self.$inner.create_dir(path).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [create_dir_all $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn create_dir_all(&self, path: &str) -> $crate::FsResult<()> {
                self.$inner.create_dir_all(path).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [copy $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn copy(&self, from: &str, to: &str) -> $crate::FsResult<()> {
                self.$inner.copy(from, to).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [move_file $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn move_file(&self, from: &str, to: &str) -> $crate::FsResult<()> {
                self.$inner.move_file(from, to).await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [sync $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn sync(&self) -> $crate::FsResult<()> {
                self.$inner.sync().await
            }
        ], [$($rest),*]);
    };
    (@munch $ty:ty, $inner:ident, [$($acc:tt)*], [open_file $(, $rest:ident)*]) => {
        $crate::forward_filesystem_impl!(@munch $ty, $inner, [$($acc)*
            async fn open_file(
                &self,
                path: &str,
                create: bool,
            ) -> $crate::FsResult<Box<dyn $crate::FilesystemFile>> {
                self.$inner.open_file(path, create).await
            }
        ], [$($rest),*]);
    };
}
