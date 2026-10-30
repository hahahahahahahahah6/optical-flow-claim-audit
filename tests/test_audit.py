import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from optical_flow_claim_audit import __version__
from optical_flow_claim_audit.audit import audit, claims, has_evidence, main, prose, FLOW_RULE, NOFLOW_RULE
from optical_flow_claim_audit.phrases import FLOW, NOFLOW

FIXTURES = Path(__file__).parent / 'fixtures'


class ClaimTests(unittest.TestCase):
    def test_backtick_fences(self):
        self.assertEqual(claims('```text\noptical flow\n```'), (False, False))

    def test_tilde_fences(self):
        self.assertEqual(claims('   ~~~~\noptical flow\n   ~~~~'), (False, False))

    def test_longer_fence_close(self):
        self.assertEqual(claims('```\noptical flow\n````\nordinary'), (False, False))

    def test_short_fence_does_not_close(self):
        self.assertEqual(claims('````\n```\noptical flow'), (False, False))

    def test_wrong_fence_does_not_close(self):
        self.assertEqual(claims('```\n~~~\noptical flow'), (False, False))

    def test_inline(self):
        self.assertEqual(claims('Provides `optical flow`.'), (True, False))

    def test_negative_removal(self):
        self.assertEqual(claims('Does not estimate optical flow.'), (False, True))

    def test_both(self):
        self.assertEqual(claims('No optical flow. But computes motion vectors.'), (True, True))

    def test_hyphens_case(self):
        self.assertEqual(claims('PER-PIXEL MOTION-VECTORS'), (True, False))

    def test_contraction(self):
        self.assertEqual(claims("Doesn't estimate optical flow."), (False, True))

    def test_curly_contraction(self):
        self.assertEqual(claims('Doesn’t estimate optical flow.'), (False, True))

    def test_wrapped_paragraph(self):
        self.assertEqual(claims('motion vector\nfield estimation'), (True, False))

    def test_paragraph_boundary(self):
        self.assertEqual(claims('optical\n\nflow'), (False, False))

    def test_word_boundaries(self):
        self.assertEqual(claims('xoptical flows'), (False, False))

    def test_longest_first(self):
        self.assertEqual(claims('does not support dense optical flow estimation'), (False, True))

    def test_phrase_sort_and_unique(self):
        for phrases in (FLOW, NOFLOW):
            self.assertEqual(list(phrases), sorted(phrases, key=len, reverse=True))
            self.assertEqual(len(phrases), len(set(phrases)))
        self.assertEqual(len(FLOW), 49)
        self.assertEqual(len(NOFLOW), 29)


POSITIVE = {
    'farneback_direct': 'import cv2\ncv2.calcOpticalFlowFarneback(a,b)',
    'farneback_alias': 'import cv2 as vision\nvision.calcOpticalFlowFarneback(a,b)',
    'farneback_from': 'from cv2 import calcOpticalFlowFarneback\ncalcOpticalFlowFarneback(a,b)',
    'farneback_from_alias': 'from cv2 import calcOpticalFlowFarneback as run\nrun(a,b)',
    'pyrlk_direct': 'import cv2\ncv2.calcOpticalFlowPyrLK(a,b)',
    'pyrlk_from': 'from cv2 import calcOpticalFlowPyrLK\ncalcOpticalFlowPyrLK(a,b)',
    'dis_direct': 'import cv2\ncv2.DISOpticalFlow_create()',
    'dis_alias': 'import cv2 as c\nc.DISOpticalFlow_create()',
    'farneback_object': 'import cv2\ncv2.FarnebackOpticalFlow_create()',
    'dualtvl1': 'import cv2\ncv2.optflow.DualTVL1OpticalFlow_create()',
    'sparse_object': 'import cv2\ncv2.SparsePyrLKOpticalFlow_create()',
    'variational': 'import cv2\ncv2.VariationalRefinement_create()',
    'cuda': 'import cv2\ncv2.cuda.SparsePyrLKOpticalFlow_create()',
    'cuda_from': 'from cv2.cuda import SparsePyrLKOpticalFlow_create as create\ncreate()',
    'raft_large_direct': 'import torchvision.models.optical_flow\ntorchvision.models.optical_flow.raft_large()',
    'raft_small_direct': 'import torchvision.models.optical_flow as motion\nmotion.raft_small()',
    'raft_large_from': 'from torchvision.models.optical_flow import raft_large\nraft_large()',
    'raft_small_from': 'from torchvision.models.optical_flow import raft_small as model\nmodel()',
    'raft_module_from': 'from torchvision.models import optical_flow as motion\nmotion.raft_large()',
    'nested_function': 'import cv2\ndef run():\n    return cv2.DISOpticalFlow_create()',
    'nested_import': 'def run():\n    import cv2 as v\n    return v.DISOpticalFlow_create()',
    'unrelated_scope': 'import cv2\ndef unrelated(cv2):\n    pass\ncv2.DISOpticalFlow_create()',
}
NEGATIVE = {
    'bare_import': 'import cv2\nfrom torchvision.models.optical_flow import raft_large',
    'image_cv': 'import cv2\ncv2.imread("a.png")',
    'image_pil': 'from PIL import Image\nImage.open("a.png")',
    'camera': 'import cv2\ncv2.VideoCapture(0).read()',
    'drawing': 'import cv2\ncv2.rectangle(image,a,b,color)',
    'detection': 'from torchvision.models.detection import fasterrcnn_resnet50_fpn\nfasterrcnn_resnet50_fpn()',
    'yolo': 'from ultralytics import YOLO\nYOLO("model").predict(image)',
    'object_detector': 'from mediapipe.tasks.python.vision import ObjectDetector\nObjectDetector.create_from_options(options)',
    'predictor': 'from detectron2.engine import DefaultPredictor\nDefaultPredictor(cfg)(image)',
    'segmentation': 'import mediapipe as mp\nmp.solutions.selfie_segmentation.SelfieSegmentation()',
    'segmenter': 'from mediapipe.tasks.python.vision import ImageSegmenter\nImageSegmenter.create_from_options(options)',
    'classification': 'from torchvision.models import resnet50\nresnet50()',
    'timm': 'import timm\ntimm.create_model("resnet50")',
    'keras': 'from keras.applications import ResNet50\nResNet50()',
    'classification_pipeline': 'from transformers import pipeline\npipeline("image-classification")',
    'face': 'import cv2\ncv2.CascadeClassifier("face.xml").detectMultiScale(image)',
    'hand': 'import mediapipe as mp\nmp.solutions.hands.Hands()',
    'pose': 'import mediapipe as mp\nmp.solutions.pose.Pose()',
    'tracking': 'import cv2\ncv2.TrackerCSRT_create()',
    'track_yolo': 'from ultralytics import YOLO\nYOLO("model").track(video)',
    'bytetrack': 'from trackers import ByteTrack, DeepSort, SORT\nByteTrack()\nDeepSort()\nSORT()',
    'dlib': 'import dlib\ndlib.correlation_tracker()',
    'depth': 'from transformers import DPTForDepthEstimation\nDPTForDepthEstimation.from_pretrained("model")',
    'depth_pipeline': 'from transformers import pipeline\npipeline("depth-estimation")',
    'depth_hub': 'import torch\ntorch.hub.load("intel-isl/MiDaS", "DPT_Large")',
    'stereo': 'import cv2\ncv2.StereoSGBM_create()',
    'ocr': 'import easyocr\neasyocr.Reader(["en"]).readtext(image)',
    'barcode': 'from pyzbar.pyzbar import decode\ndecode(image)',
    'diff': 'import cv2\ncv2.absdiff(a,b)\nc = a-b',
    'video': 'import cv2\ncv2.VideoWriter("out.avi",codec,30,size)',
    'kalman': 'import cv2\ncv2.KalmanFilter(4,2)',
    'local_function': 'def calcOpticalFlowFarneback(a,b): pass\ncalcOpticalFlowFarneback(a,b)',
    'shadow_from': 'from cv2 import calcOpticalFlowFarneback\ndef calcOpticalFlowFarneback(a,b): pass\ncalcOpticalFlowFarneback(a,b)',
    'shadow_module': 'import cv2\nclass cv2: pass\ncv2.DISOpticalFlow_create()',
    'shadow_parameter': 'import cv2\ndef run(cv2):\n    cv2.DISOpticalFlow_create()',
    'shadow_assignment': 'import cv2\ncv2 = fake\ncv2.DISOpticalFlow_create()',
    'shadow_local': 'import cv2\ndef run():\n    cv2.DISOpticalFlow_create()\n    cv2 = fake',
    'shadow_lambda': 'import cv2\nx = lambda cv2: cv2.DISOpticalFlow_create()',
    'wrong_provider': 'from fake import calcOpticalFlowFarneback\ncalcOpticalFlowFarneback(a,b)',
    'wrong_raft': 'from models import raft_large\nraft_large()',
    'string': 'x = "cv2.calcOpticalFlowFarneback(a,b) optical flow"',
    'subprocess': 'import subprocess\nsubprocess.run(["optical-flow", "image.png"])',
    'invalid': 'import cv2\ncv2.DISOpticalFlow_create()\ndef broken(',
    'relative_import': 'from .cv2 import DISOpticalFlow_create\nDISOpticalFlow_create()',
}


class EvidenceTests(unittest.TestCase):
    pass


def evidence_test(source, expected):
    def test(self):
        self.assertEqual(has_evidence(source), expected)
    return test


for name, source in POSITIVE.items():
    setattr(EvidenceTests, 'test_' + name, evidence_test(source, True))
for name, source in NEGATIVE.items():
    setattr(EvidenceTests, 'test_' + name, evidence_test(source, False))
for family, phrases in [('flow', FLOW), ('noflow', NOFLOW)]:
    for index, phrase in enumerate(phrases):
        def phrase_test(self, phrase=phrase, family=family):
            self.assertEqual(claims(phrase), (family == 'flow', family == 'noflow'))
        setattr(ClaimTests, 'test_template_' + family + '_' + str(index), phrase_test)


class IntegrationTests(unittest.TestCase):
    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(map(str, args)))
        return code, out.getvalue(), err.getvalue()

    def test_good(self):
        self.assertEqual(audit(FIXTURES / 'good')['findings'], [])
        self.assertEqual(self.cli(FIXTURES / 'good')[0], 0)

    def test_bad(self):
        result = audit(FIXTURES / 'bad')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual([(f['severity'], f['rule']) for f in result['findings']], [('HIGH', FLOW_RULE)])
        self.assertEqual(self.cli(FIXTURES / 'bad')[0], 2)

    def test_testonly(self):
        result = audit(FIXTURES / 'testonly')
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual([(f['severity'], f['rule']) for f in result['findings']], [('MEDIUM', FLOW_RULE)])
        self.assertEqual(self.cli(FIXTURES / 'testonly')[0], 0)
        self.assertEqual(self.cli(FIXTURES / 'testonly', '--fail-on', 'MEDIUM')[0], 2)

    def test_negative(self):
        result = audit(FIXTURES / 'no_flow')
        self.assertEqual([(f['severity'], f['rule']) for f in result['findings']], [('HIGH', NOFLOW_RULE)])
        self.assertEqual(self.cli(FIXTURES / 'no_flow')[0], 2)

    def test_no_claim(self):
        self.assertEqual(audit(FIXTURES / 'no_claim')['verdict'], 'PASS')

    def test_ignore(self):
        self.assertEqual(audit(FIXTURES / 'bad', ['README.*'])['findings'], [])
        self.assertEqual(audit(FIXTURES / 'good', ['*.py'])['findings'][0]['severity'], 'HIGH')

    def test_json(self):
        code, output, _ = self.cli(FIXTURES / 'bad', '--json')
        self.assertEqual(code, 2)
        result = json.loads(output)
        self.assertEqual(set(result), {'verdict', 'findings', 'files_scanned'})
        self.assertEqual(set(result['findings'][0]), {'rule', 'severity', 'message'})
        self.assertEqual(result['files_scanned'], 2)

    def test_verbose(self):
        self.assertIn('Files scanned: 2', self.cli(FIXTURES / 'good', '--verbose')[1])

    def test_human_finding(self):
        self.assertIn('HIGH ' + FLOW_RULE + ': ', self.cli(FIXTURES / 'bad')[1])

    def test_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as ctx:
            main(['--version'])
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(output.getvalue(), '0.1.0\n')
        self.assertEqual(__version__, '0.1.0')

    def test_invalid_root(self):
        self.assertEqual(self.cli(FIXTURES / 'missing')[0], 1)
        self.assertEqual(self.cli(FIXTURES / 'good' / 'app.py')[0], 1)

    def test_usage_errors(self):
        for args in ([], ['--unknown'], [str(FIXTURES), '--fail-on', 'LOW']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                main(args)
            self.assertEqual(ctx.exception.code, 1)

    def test_module_entry(self):
        result = subprocess.run([sys.executable, '-m', 'optical_flow_claim_audit', str(FIXTURES / 'good'), '--json'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)['verdict'], 'PASS')

    def test_project_self_audit(self):
        project = Path(__file__).resolve().parents[1]
        self.assertEqual(audit(project)['findings'], [])
        text = (project / 'README.md').read_text()
        self.assertEqual(claims(text), (False, False))
        normalized = ' '.join(prose(text).lower().replace('-', ' ').split())
        for phrase in FLOW + NOFLOW:
            self.assertNotIn(phrase, normalized)


class FileSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('README.md', 'Optical flow')

    def write(self, name, text):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def test_excluded_components(self):
        for component in ('fixtures', 'docs', 'examples'):
            self.write(component + '/app.py', POSITIVE['dis_direct'])
        self.assertEqual(audit(self.root)['findings'][0]['severity'], 'HIGH')

    def test_test_filename_and_component(self):
        for path in ('tests/helper.py', 'test_model.py', 'model_test.py'):
            p = self.write(path, POSITIVE['dis_direct'])
            self.assertEqual(audit(self.root)['findings'][0]['severity'], 'MEDIUM')
            p.unlink()

    def test_other_markdown_ignored(self):
        (self.root / 'README.md').unlink()
        self.write('notes.md', 'Optical flow')
        self.assertEqual(audit(self.root)['findings'], [])

    def test_nested_docs(self):
        (self.root / 'README.md').unlink()
        self.write('pkg/docs/deep/page.md', 'Optical flow')
        self.assertEqual(len(audit(self.root)['findings']), 1)

    def test_symlink_file(self):
        target = self.write('examples/target.py', POSITIVE['dis_direct'])
        (self.root / 'linked.py').symlink_to(target)
        self.assertEqual(audit(self.root)['findings'][0]['severity'], 'HIGH')

    def test_symlink_directory(self):
        target = self.write('examples/code/target.py', POSITIVE['dis_direct']).parent
        (self.root / 'linked').symlink_to(target, target_is_directory=True)
        self.assertEqual(audit(self.root)['findings'][0]['severity'], 'HIGH')

    def test_directory_ignore(self):
        self.write('vendor/app.py', POSITIVE['dis_direct'])
        self.assertEqual(audit(self.root, ['vendor'])['findings'][0]['severity'], 'HIGH')

    def test_multiple_claims_one_finding(self):
        self.write('docs/page.md', 'Optical flow\n\nComputes motion vectors')
        self.assertEqual(len(audit(self.root)['findings']), 1)

    def test_imports_do_not_cross_modules(self):
        self.write('one.py', 'import cv2')
        self.write('two.py', 'cv2.DISOpticalFlow_create()')
        self.assertEqual(audit(self.root)['findings'][0]['severity'], 'HIGH')

    def test_syntax_error_no_evidence(self):
        self.write('broken.py', NEGATIVE['invalid'])
        self.assertEqual(audit(self.root)['findings'][0]['severity'], 'HIGH')

    def test_negative_test_evidence_only(self):
        self.write('README.md', 'No optical flow')
        self.write('test_model.py', POSITIVE['dis_direct'])
        self.assertEqual(audit(self.root)['findings'], [])

    def test_unreadable_root(self):
        self.root.chmod(0o000)
        try:
            with self.assertRaises(ValueError):
                audit(self.root)
        finally:
            self.root.chmod(0o700)

    def test_bad_utf8(self):
        (self.root / 'bad.py').write_bytes(b'\xff')
        with self.assertRaises(ValueError):
            audit(self.root)


if __name__ == '__main__':
    unittest.main()
