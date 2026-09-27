# Demo signing keys

The HMAC demo signing keys that used to live here were removed before this
repository was made public. In HMAC fallback mode the `.pub` file is a copy of
the same shared secret, so both files were secret material.

You do not need these keys to run the demo: `python -m cviaf demo` generates a
fresh keypair into its output directory on every run.

To restore the original demo keys locally (only needed to re-verify the seals
in the historical `demo-output/` / `cviaf_demo_output/` reports): ask the team
for the vault-held base64 values, then for each key id:

```bash
echo '<base64-from-vault>' | base64 -d > keys/<key_id>.priv
cp keys/<key_id>.priv keys/<key_id>.pub   # identical in HMAC mode
```

`*.priv` and the output directories are gitignored, so restored keys stay local.
