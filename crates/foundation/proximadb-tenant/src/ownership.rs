// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! The ONE rule for which request tenant may resolve a stored collection.
//!
//! Every collection-resolution point that serves a request (search, get/scan,
//! collection GET/list) calls [`collection_visible_to`]; none restates the
//! predicate. Ownership is read from the collection's persisted config:
//!
//! * the `tenant:<id>` tag, and
//! * the `owner` field when the collection is flagged `tenant_isolated:true`.
//!
//! A collection with neither is *unattributed* (created before ownership was
//! recorded) and belongs to [`crate::DEFAULT_TENANT`]: single-tenant
//! deployments, whose requests all resolve to the default tenant, see no
//! change, and no other tenant can resolve it. When the signals disagree the
//! collection is *conflicting*: no request tenant may resolve it (fail closed),
//! and the conflict is logged.
//!
//! Tenant ids are compared exactly after trimming surrounding whitespace — no
//! case folding (`Acme` and `acme` are different tenants).

use proximadb_proto::proximadb_v1::Collection;

/// Tag prefix that records a collection's owning tenant.
pub const OWNER_TAG_PREFIX: &str = "tenant:";
/// Tag marking a collection whose `owner` field is its owning tenant.
pub const TENANT_ISOLATED_TAG: &str = "tenant_isolated:true";

/// What a collection's persisted config says about its owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectionOwner {
    /// Exactly one tenant is recorded (tag and/or isolated owner field agree).
    Tenant(String),
    /// Nothing recorded: owned by [`crate::DEFAULT_TENANT`].
    Unattributed,
    /// The recorded signals name different tenants. Resolvable by no one.
    Conflicting(Vec<String>),
}

/// Read the owner recorded on `collection`.
pub fn collection_owner(collection: &Collection) -> CollectionOwner {
    let Some(config) = collection.config.as_ref() else {
        return CollectionOwner::Unattributed;
    };
    let mut recorded: Vec<String> = config
        .tags
        .iter()
        .filter_map(|tag| tag.strip_prefix(OWNER_TAG_PREFIX))
        .map(str::trim)
        .filter(|tenant| !tenant.is_empty())
        .map(str::to_string)
        .collect();
    if config.tags.iter().any(|tag| tag == TENANT_ISOLATED_TAG)
        && let Some(owner) = config
            .owner
            .as_deref()
            .map(str::trim)
            .filter(|owner| !owner.is_empty())
    {
        recorded.push(owner.to_string());
    }
    recorded.sort();
    recorded.dedup();
    match recorded.len() {
        0 => CollectionOwner::Unattributed,
        1 => CollectionOwner::Tenant(recorded.remove(0)),
        _ => CollectionOwner::Conflicting(recorded),
    }
}

/// The tenant a collection belongs to for request scoping, or `None` when its
/// recorded ownership is conflicting (fail closed).
pub fn effective_owner(collection: &Collection) -> Option<String> {
    match collection_owner(collection) {
        CollectionOwner::Tenant(tenant) => Some(tenant),
        CollectionOwner::Unattributed => Some(crate::DEFAULT_TENANT.to_string()),
        CollectionOwner::Conflicting(tenants) => {
            tracing::error!(
                collection = %collection.id,
                recorded = ?tenants,
                "collection records conflicting owners; it is not resolvable by any tenant \
                 until its owner tags are corrected"
            );
            None
        }
    }
}

/// THE scoping rule: may a request acting as `request_tenant` (raw signal;
/// absent/blank resolves to the default tenant) resolve `collection`?
pub fn collection_visible_to(collection: &Collection, request_tenant: Option<&str>) -> bool {
    let request = crate::resolve_request_tenant(request_tenant);
    effective_owner(collection).is_some_and(|owner| owner == request)
}

/// Record `tenant` as the owner of a collection config about to be created:
/// replaces any owner signals the caller supplied (the owner is the creating
/// tenant, never a client-chosen value).
pub fn stamp_collection_owner(
    config: &mut proximadb_proto::proximadb_v1::CollectionConfig,
    tenant: &str,
) {
    let tenant = crate::resolve_request_tenant(Some(tenant));
    config
        .tags
        .retain(|tag| !tag.starts_with(OWNER_TAG_PREFIX) && tag != TENANT_ISOLATED_TAG);
    config.tags.push(format!("{OWNER_TAG_PREFIX}{tenant}"));
    config.tags.push(TENANT_ISOLATED_TAG.to_string());
    config.owner = Some(tenant);
}

/// Keep a collection's recorded owner signals across a config update: the
/// owner tags and the `owner` field of `updated` are reset to exactly those of
/// `previous`. Ownership changes only through [`stamp_collection_owner`] at
/// create or the owner backfill, never through a client update.
pub fn carry_over_owner(
    previous: &proximadb_proto::proximadb_v1::CollectionConfig,
    updated: &mut proximadb_proto::proximadb_v1::CollectionConfig,
) {
    let is_owner_tag =
        |tag: &String| tag.starts_with(OWNER_TAG_PREFIX) || tag == TENANT_ISOLATED_TAG;
    updated.tags.retain(|tag| !is_owner_tag(tag));
    updated.tags.extend(
        previous
            .tags
            .iter()
            .filter(|tag| is_owner_tag(tag))
            .cloned(),
    );
    updated.owner = previous.owner.clone();
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_proto::proximadb_v1::CollectionConfig;

    fn collection(tags: &[&str], owner: Option<&str>) -> Collection {
        Collection {
            id: "7".to_string(),
            config: Some(CollectionConfig {
                name: "docs".to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                owner: owner.map(str::to_string),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn owner_from_tag() {
        let c = collection(&["tenant:acme"], None);
        assert_eq!(collection_owner(&c), CollectionOwner::Tenant("acme".into()));
        assert!(collection_visible_to(&c, Some("acme")));
        assert!(!collection_visible_to(&c, Some("globex")));
        assert!(
            !collection_visible_to(&c, None),
            "default may not read acme's"
        );
    }

    #[test]
    fn owner_from_isolated_owner_field() {
        let c = collection(&["tenant_isolated:true"], Some("acme"));
        assert_eq!(collection_owner(&c), CollectionOwner::Tenant("acme".into()));
        assert!(collection_visible_to(&c, Some("acme")));
        assert!(!collection_visible_to(&c, Some("globex")));
    }

    #[test]
    fn owner_field_without_isolation_flag_is_not_ownership() {
        // `owner` alone may be a user/display value; only the flag makes it a tenant.
        let c = collection(&[], Some("acme"));
        assert_eq!(collection_owner(&c), CollectionOwner::Unattributed);
        assert!(collection_visible_to(&c, None));
        assert!(!collection_visible_to(&c, Some("acme")));
    }

    #[test]
    fn unattributed_belongs_to_default_only() {
        for c in [
            collection(&[], None),
            collection(&["tenant:", "tenant:   "], None),
            Collection::default(),
        ] {
            assert_eq!(collection_owner(&c), CollectionOwner::Unattributed);
            assert!(collection_visible_to(&c, None));
            assert!(collection_visible_to(&c, Some("")));
            assert!(collection_visible_to(&c, Some(" default ")));
            assert!(!collection_visible_to(&c, Some("acme")));
        }
    }

    #[test]
    fn whitespace_is_trimmed_and_case_is_significant() {
        let c = collection(&["tenant: acme "], None);
        assert!(collection_visible_to(&c, Some("acme")));
        assert!(collection_visible_to(&c, Some("  acme\t")));
        assert!(!collection_visible_to(&c, Some("Acme")));
        assert!(!collection_visible_to(&c, Some("ACME")));
    }

    #[test]
    fn agreeing_tag_and_field_are_one_owner() {
        let c = collection(&["tenant:acme", "tenant_isolated:true"], Some("acme"));
        assert_eq!(collection_owner(&c), CollectionOwner::Tenant("acme".into()));
        assert!(collection_visible_to(&c, Some("acme")));
    }

    #[test]
    fn mixed_signals_fail_closed_for_everyone() {
        for c in [
            collection(&["tenant:acme", "tenant_isolated:true"], Some("globex")),
            collection(&["tenant:acme", "tenant:globex"], None),
        ] {
            assert!(matches!(
                collection_owner(&c),
                CollectionOwner::Conflicting(ref t) if t.len() == 2
            ));
            for tenant in [None, Some("acme"), Some("globex"), Some("default")] {
                assert!(
                    !collection_visible_to(&c, tenant),
                    "{tenant:?} resolved a conflict"
                );
            }
        }
    }

    #[test]
    fn stamping_replaces_client_supplied_owner_signals() {
        let mut config = CollectionConfig {
            tags: vec![
                "env:prod".into(),
                "tenant:globex".into(),
                "tenant_isolated:true".into(),
            ],
            owner: Some("globex".into()),
            ..Default::default()
        };
        stamp_collection_owner(&mut config, " acme ");
        let c = Collection {
            config: Some(config.clone()),
            ..Default::default()
        };
        assert_eq!(collection_owner(&c), CollectionOwner::Tenant("acme".into()));
        assert!(config.tags.contains(&"env:prod".to_string()));
        assert_eq!(config.owner.as_deref(), Some("acme"));
        let mut blank = CollectionConfig::default();
        stamp_collection_owner(&mut blank, "");
        assert_eq!(blank.owner.as_deref(), Some(crate::DEFAULT_TENANT));
    }

    #[test]
    fn update_cannot_change_the_recorded_owner() {
        let previous = CollectionConfig {
            tags: vec![
                "tenant:acme".into(),
                "tenant_isolated:true".into(),
                "a".into(),
            ],
            owner: Some("acme".into()),
            ..Default::default()
        };
        let mut updated = CollectionConfig {
            tags: vec!["tenant:globex".into(), "b".into()],
            owner: Some("globex".into()),
            ..Default::default()
        };
        carry_over_owner(&previous, &mut updated);
        let c = Collection {
            config: Some(updated.clone()),
            ..Default::default()
        };
        assert_eq!(collection_owner(&c), CollectionOwner::Tenant("acme".into()));
        assert!(updated.tags.contains(&"b".to_string()));
        assert_eq!(updated.owner.as_deref(), Some("acme"));
        // An unattributed collection stays unattributed through an update.
        let mut claimed = CollectionConfig {
            tags: vec!["tenant:globex".into()],
            ..Default::default()
        };
        carry_over_owner(&CollectionConfig::default(), &mut claimed);
        let c = Collection {
            config: Some(claimed),
            ..Default::default()
        };
        assert_eq!(collection_owner(&c), CollectionOwner::Unattributed);
    }
}
