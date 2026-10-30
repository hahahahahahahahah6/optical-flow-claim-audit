"""Offline documentation and Python AST inspection; target code is never run."""
import argparse
import ast
import fnmatch
import json
import os
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import FLOW, NOFLOW

FLOW_RULE = 'documented-optical-flow-claim-false'
NOFLOW_RULE = 'documented-no-optical-flow-claim-false'
CV_CALLS = {
    'calcOpticalFlowFarneback', 'calcOpticalFlowPyrLK',
    'DISOpticalFlow_create', 'FarnebackOpticalFlow_create',
    'DualTVL1OpticalFlow_create', 'SparsePyrLKOpticalFlow_create',
    'VariationalRefinement_create',
}


def prose(text):
    """Remove CommonMark-style backtick and tilde fenced blocks."""
    lines = []
    fence = None
    for line in text.splitlines():
        if fence:
            if re.fullmatch(r' {0,3}' + re.escape(fence[0]) + '{' + str(fence[1]) + r',}\s*', line):
                fence = None
                lines.append('')
            continue
        match = re.match(r' {0,3}(`{3,}|~{3,})(.*)$', line)
        if match and not (match[1][0] == '`' and '`' in match[2]):
            fence = (match[1][0], len(match[1]))
            lines.append('')
        else:
            lines.append(line)
    return '\n'.join(lines)


def normalize(text):
    return re.sub(r'\s+', ' ', text.lower().replace('’', "'").replace("n't", ' not').replace('-', ' ')).strip()


def _pattern(phrases):
    return re.compile(r'\b(?:' + '|'.join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + r')\b')


def claims(text):
    """Return (positive, negative), keeping paragraph boundaries distinct."""
    positive = negative = False
    for paragraph in re.split(r'\n\s*\n', prose(text)):
        paragraph = normalize(paragraph)
        pattern = _pattern(NOFLOW)
        negative = negative or bool(pattern.search(paragraph))
        remaining = pattern.sub(' ', paragraph)
        positive = positive or bool(_pattern(FLOW).search(remaining))
    return positive, negative


def _scope_bindings(node):
    """Collect bindings without leaking child scopes into their parent."""
    bindings = {}

    class Bindings(ast.NodeVisitor):
        def visit_Import(self, item):
            for alias in item.names:
                bindings[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]

        def visit_ImportFrom(self, item):
            for alias in item.names:
                if alias.name != '*':
                    bindings[alias.asname or alias.name] = ((item.module + '.' + alias.name) if item.module and not item.level else None)

        def visit_Name(self, item):
            if isinstance(item.ctx, (ast.Store, ast.Del)):
                bindings[item.id] = None

        def visit_FunctionDef(self, item):
            bindings[item.name] = None

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        def visit_Lambda(self, item):
            pass

    collector = Bindings()
    for child in node.body:
        collector.visit(child)
    # Definitions shadow imports even when textually earlier, conservatively.
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[child.name] = None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            bindings[arg.arg] = None
        for arg in (args.vararg, args.kwarg):
            if arg:
                bindings[arg.arg] = None
    return bindings


def has_evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    found = False

    class Evidence(ast.NodeVisitor):
        def __init__(self):
            self.scopes = [_scope_bindings(tree)]

        def resolve(self, node):
            if isinstance(node, ast.Name):
                for scope in reversed(self.scopes):
                    if node.id in scope:
                        return scope[node.id]
            if isinstance(node, ast.Attribute):
                base = self.resolve(node.value)
                if base:
                    return base + '.' + node.attr
            return None

        def visit_Call(self, node):
            nonlocal found
            path = self.resolve(node.func)
            if path:
                bits = path.split('.')
                if (bits[0] == 'cv2' and bits[-1] in CV_CALLS) or (bits[-1] in {'raft_large', 'raft_small'} and 'optical_flow' in bits[:-1]):
                    found = True
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            for expr in node.decorator_list + node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                self.visit(expr)
            self.scopes.append(_scope_bindings(node))
            for child in node.body:
                self.visit(child)
            self.scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            for expr in node.bases + node.decorator_list:
                self.visit(expr)
            self.scopes.append(_scope_bindings(node))
            for child in node.body:
                self.visit(child)
            self.scopes.pop()

        def visit_Lambda(self, node):
            args = node.args
            names = {a.arg: None for a in args.posonlyargs + args.args + args.kwonlyargs}
            for a in (args.vararg, args.kwarg):
                if a:
                    names[a.arg] = None
            self.scopes.append(names)
            self.visit(node.body)
            self.scopes.pop()

    Evidence().visit(tree)
    return found


def audit(root, ignore=(), fail_on='HIGH'):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('root must be a readable, non-symlink directory')
    if not os.access(root, os.R_OK | os.X_OK) or root.stat().st_mode & 0o444 == 0:
        raise ValueError('root is unreadable')
    positive = negative = production = test = False
    scanned = 0

    def walk_error(error):
        raise ValueError('cannot read project: ' + str(error))

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink() and not any(fnmatch.fnmatchcase((Path(directory) / d).relative_to(root).as_posix(), g) for g in ignore))
        for name in sorted(files):
            path = Path(directory) / name
            rel = path.relative_to(root)
            if path.is_symlink() or any(fnmatch.fnmatchcase(rel.as_posix(), g) for g in ignore):
                continue
            doc = rel.as_posix() == 'README.md' or (path.suffix == '.md' and 'docs' in rel.parts[:-1])
            source = path.suffix == '.py' and not {'fixtures', 'docs', 'examples'}.intersection(rel.parts[:-1])
            if not (doc or source):
                continue
            try:
                content = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as error:
                raise ValueError('cannot read ' + rel.as_posix() + ': ' + str(error)) from error
            scanned += 1
            if doc:
                pos, neg = claims(content)
                positive |= pos
                negative |= neg
            if source and has_evidence(content):
                is_test = 'tests' in rel.parts[:-1] or name.startswith('test_') or name.endswith('_test.py')
                if is_test:
                    test = True
                else:
                    production = True
    findings = []
    if positive and not production:
        findings.append({'rule': FLOW_RULE, 'severity': 'MEDIUM' if test else 'HIGH', 'message': 'Documented pixel motion capability has no production provider call.'})
    if negative and production:
        findings.append({'rule': NOFLOW_RULE, 'severity': 'HIGH', 'message': 'Documented absence conflicts with a production provider call.'})
    failed = any(f['severity'] == 'HIGH' or fail_on == 'MEDIUM' for f in findings)
    return {'verdict': 'FAIL' if failed else 'PASS', 'findings': findings, 'files_scanned': scanned}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(description='Audit documented pixel motion capabilities against Python source.')
    parser.add_argument('root')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore', action='append', default=[], metavar='GLOB')
    parser.add_argument('--fail-on', choices=['HIGH', 'MEDIUM'], default='HIGH')
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    try:
        result = audit(args.root, args.ignore, args.fail_on)
    except (ValueError, OSError) as error:
        print('error: ' + str(error), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result))
    else:
        print(result['verdict'])
        for finding in result['findings']:
            print('{severity} {rule}: {message}'.format(**finding))
        if args.verbose:
            print('Files scanned: ' + str(result['files_scanned']))
    return 2 if result['verdict'] == 'FAIL' else 0
