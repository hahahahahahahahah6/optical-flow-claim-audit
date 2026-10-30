# Pixel motion claim audit

An offline, dependency-free static checker for documented pixel motion fields
and motion between frames. Requires Python 3.9 or newer. It reads Python syntax
without importing or executing the project under review.

## Install and run

```sh
python3 -m pip install optical-flow-claim-audit
optical-flow-claim-audit ./project --json
python3 -m optical_flow_claim_audit ./project --verbose
optical-flow-claim-audit ./project --ignore 'vendor/*' --fail-on MEDIUM
optical-flow-claim-audit --version
```

The root README.md and Markdown files beneath any directory named docs supply
claims. Backtick and tilde fenced blocks are excluded; inline code counts.
Paragraphs join wrapped lines. Matching ignores case, expands negative
contractions, replaces hyphens with spaces, and uses exact templates with word
boundaries, longest first. Negative spans are removed before positive matching.
The complete template tuples are in the package's phrases.py module.

## Evidence and scope

Provider-resolved calls to the following OpenCV operations count, including
import aliases and direct imports. Construction alone counts for provider
objects and the two RAFT models. RAFT imports must include the designated
provider module component shown below.

```text
cv2.calcOpticalFlowFarneback
cv2.calcOpticalFlowPyrLK
cv2.DISOpticalFlow_create
cv2.FarnebackOpticalFlow_create
cv2.DualTVL1OpticalFlow_create
cv2.SparsePyrLKOpticalFlow_create
cv2.VariationalRefinement_create
cv2.cuda forms of these constructors
torchvision.models.optical_flow.raft_large
torchvision.models.optical_flow.raft_small
```

Bare imports, image loading, camera capture, drawing, detection, segmentation,
classification, face/hand/pose estimation, identity tracking, depth estimation,
OCR, barcode decoding, frame differencing, video writing, Kalman filters,
strings, subprocess commands, and local lookalikes do not count.

Python files beneath fixtures, docs, and examples are excluded from evidence.
Files beneath tests, or named with a test_ prefix or _test.py suffix, supply
only test evidence. Syntax-invalid files supply no evidence. Symlinked files
and directories are skipped. Repeatable ignore globs match relative POSIX paths
using fnmatch; matching directories are pruned. The scanned count includes only
selected documentation and source files actually read.

## Results and exits

At most one finding is emitted per claim family:

```text
documented-optical-flow-claim-false
  HIGH: positive claim without production or test evidence
  MEDIUM: positive claim with evidence only in tests
documented-no-optical-flow-claim-false
  HIGH: negative claim conflicts with production evidence
```

No claim means PASS. By default only HIGH findings fail the audit; the MEDIUM
threshold also fails on test-only findings. Exit 0 means PASS, 2 means FAIL,
and 1 means an invalid or unreadable root or a command usage error. A PASS can
include findings below the selected threshold. Human output contains a verdict
and one line per finding; verbose mode adds the scanned count. JSON contains
verdict, findings (rule, severity, message), and files_scanned.

## Limitations and development

This is a syntax fingerprint check, not a correctness proof. It does not trace
values across modules, resolve dynamic imports or assignments to callable
aliases, establish reachability, validate model outputs, or prove that two
frames are used correctly. Calls in unreachable code may count. Local bindings
shadow imports conservatively within their scopes. Unsupported providers can
produce missing-evidence findings; documentation paraphrases outside the exact
templates are intentionally ignored. Unreadable selected files are errors.

```sh
python3 -m unittest discover
python3 -m build
python3 -m optical_flow_claim_audit . --json
```

The source distribution includes the unittest suite and its small fixtures.
Distributed under the MIT license.
