#!/usr/bin/env bash
# Apply the repository settings and rulesets versioned in .github/rulesets/.
#
#   bash scripts/apply-github-settings.sh [owner/repo]
#
# The rulesets live in the repository so that branch protection is reviewed like code:
# change the JSON in a pull request, merge it, re-run this script. Idempotent -- a
# ruleset that already exists (matched by name) is updated in place.
#
# Needs the GitHub CLI authenticated as a repository admin (`gh auth login`).
set -euo pipefail

repo="${1:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "configuring ${repo}"

# Merge policy: squash or rebase only, so history on main stays linear and every commit
# on it passed CI as a unit.
gh api --method PATCH "repos/${repo}" --silent \
  -F delete_branch_on_merge=true \
  -F allow_merge_commit=false \
  -F allow_squash_merge=true \
  -F allow_rebase_merge=true \
  -F allow_auto_merge=true \
  -F has_wiki=false
echo "  merge policy: squash/rebase, delete merged branches"

# Security features. Each is attempted independently: one that the plan does not offer
# should not stop the rest.
try() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "  enabled: ${label}"
  else
    echo "  WARNING: could not enable ${label} (check the repository's plan and settings)"
  fi
}

try "secret scanning + push protection" \
  gh api --method PATCH "repos/${repo}" --input - <<'JSON'
{"security_and_analysis": {"secret_scanning": {"status": "enabled"},
                           "secret_scanning_push_protection": {"status": "enabled"}}}
JSON
try "Dependabot alerts" gh api --method PUT "repos/${repo}/vulnerability-alerts"
try "Dependabot security updates" gh api --method PUT "repos/${repo}/automated-security-fixes"
try "private vulnerability reporting" \
  gh api --method PUT "repos/${repo}/private-vulnerability-reporting"

# Rulesets.
existing="$(gh api "repos/${repo}/rulesets" --paginate --jq '.[] | "\(.id)\t\(.name)"')"
for file in "${root}"/.github/rulesets/*.json; do
  name="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "${file}")"
  id="$(printf '%s\n' "${existing}" | awk -F'\t' -v n="${name}" '$2 == n { print $1 }')"
  if [[ -n "${id}" ]]; then
    method=PUT
    endpoint="repos/${repo}/rulesets/${id}"
    verb=updated
  else
    method=POST
    endpoint="repos/${repo}/rulesets"
    verb=created
  fi
  if gh api --method "${method}" "${endpoint}" --input "${file}" --silent; then
    echo "  ${verb} ruleset: ${name}"
  else
    echo "  WARNING: ruleset '${name}' was not applied (see the error above)"
  fi
done

echo "done: https://github.com/${repo}/settings/rules"
