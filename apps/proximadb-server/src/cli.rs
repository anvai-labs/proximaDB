//! Command line. Serving stays the default — every existing flat invocation
//! (`proximadb-server -c cfg -d dir -p port`, as used by CI, the Dockerfiles
//! and `deploy/docker/entrypoint.sh`) parses exactly as before. Offline
//! maintenance is an optional subcommand that shares the same `-c`/`-d`
//! resolution, so it always targets the roots the server would own.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "proximadb-server", version)]
#[command(about = "ProximaDB cloud-native vector database server")]
pub(crate) struct Cli {
    #[command(flatten)]
    pub(crate) common: CommonArgs,

    #[command(flatten)]
    pub(crate) serve: ServeArgs,

    #[command(subcommand)]
    pub(crate) command: Option<AdminCommand>,
}

/// Options that locate the server's storage — shared by serving and by the
/// offline subcommands (`global` so they parse on either side of the verb).
#[derive(Args, Debug)]
pub(crate) struct CommonArgs {
    #[arg(short, long, default_value = "config/config.toml", global = true)]
    pub(crate) config: PathBuf,

    #[arg(short, long, global = true)]
    pub(crate) data_dir: Option<PathBuf>,

    #[arg(
        short,
        long,
        global = true,
        help = "Set log level (error, warn, info, debug, trace)"
    )]
    pub(crate) log_level: Option<String>,
}

/// Options that only make sense for a serving process.
#[derive(Args, Debug)]
pub(crate) struct ServeArgs {
    #[arg(short, long)]
    pub(crate) port: Option<u16>,

    #[arg(long)]
    pub(crate) node_id: Option<String>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum AdminCommand {
    /// Offline cold backup of every local storage root. The server must be
    /// stopped: backup takes the server's own ownership lease and refuses if
    /// it cannot.
    Backup {
        /// Destination directory (must be empty or not exist, and outside every
        /// storage root).
        #[arg(long)]
        to: PathBuf,
    },
    /// Restore a cold backup into the configured storage roots. Verifies every
    /// checksum before writing anything; the configured root layout must match
    /// the backup's.
    Restore {
        /// Backup directory written by `backup`.
        #[arg(long)]
        from: PathBuf,
        /// Replace non-empty storage roots instead of refusing.
        #[arg(long)]
        force: bool,
    },
}

impl Cli {
    /// Serving-only flags are meaningless for an offline subcommand; reject
    /// them rather than silently ignore what the operator typed.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self.command.is_some() {
            anyhow::ensure!(
                self.serve.port.is_none() && self.serve.node_id.is_none(),
                "--port/--node-id apply to serving only, not to backup/restore"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("proximadb-server").chain(args.iter().copied()))
            .expect("parse")
    }

    #[test]
    fn flat_serve_invocation_unchanged() {
        let cli = parse(&["-c", "cfg.toml", "-d", "/tmp/d", "-p", "5678"]);
        assert_eq!(cli.common.config, PathBuf::from("cfg.toml"));
        assert_eq!(cli.common.data_dir, Some(PathBuf::from("/tmp/d")));
        assert_eq!(cli.serve.port, Some(5678));
        assert!(cli.command.is_none());

        // The forms deploy/docker/entrypoint.sh rewrites and passes through.
        assert_eq!(
            parse(&["--config=x.toml"]).common.config,
            PathBuf::from("x.toml")
        );
        assert_eq!(parse(&["-cx.toml"]).common.config, PathBuf::from("x.toml"));
        assert_eq!(
            parse(&["--node-id", "n1"]).serve.node_id.as_deref(),
            Some("n1")
        );
        assert_eq!(
            parse(&[]).common.config,
            PathBuf::from("config/config.toml")
        );
    }

    #[test]
    fn backup_subcommand_parses_with_global_config_either_side() {
        for args in [
            &["-c", "cfg.toml", "backup", "--to", "/bk"][..],
            &["backup", "-c", "cfg.toml", "--to", "/bk"][..],
        ] {
            let cli = parse(args);
            assert_eq!(cli.common.config, PathBuf::from("cfg.toml"));
            assert!(matches!(
                cli.command,
                Some(AdminCommand::Backup { ref to }) if to == &PathBuf::from("/bk")
            ));
        }
    }

    #[test]
    fn restore_force_flag() {
        let cli = parse(&["restore", "--from", "/bk", "--force"]);
        assert!(matches!(
            cli.command,
            Some(AdminCommand::Restore { force: true, .. })
        ));
        let cli = parse(&["restore", "--from", "/bk"]);
        assert!(matches!(
            cli.command,
            Some(AdminCommand::Restore { force: false, .. })
        ));
    }

    #[test]
    fn subcommand_rejects_port() {
        assert!(
            parse(&["-p", "5678", "backup", "--to", "/bk"])
                .validate()
                .is_err()
        );
        assert!(parse(&["backup", "--to", "/bk"]).validate().is_ok());
        assert!(parse(&["-p", "5678"]).validate().is_ok());
    }

    #[test]
    fn version_flag_is_supported() {
        // release.yml's Homebrew test runs `proximadb-server --version`.
        let err = Cli::try_parse_from(["proximadb-server", "--version"]).expect_err("version");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
    }
}
