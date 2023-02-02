from unittest import TestCase
import logging
from ..factory import PostProcessorFactory
from ..adam import PostProcessorAdam


class TestPostProcessorFactory(TestCase):
    @classmethod
    def setUpClass(self):
        self.logger = logging.getLogger()
        self.post_processor = PostProcessorFactory()

    def setUp(self):
        self.logger.info("Test %s Started" % (self._testMethodName))

    def tearDown(self):
        self.logger.info("Test %s Finished" % (self._testMethodName))

    def test_invalid_create_post_processor(self):

        invalid_method = "dummy-method"
        with self.assertRaisesRegex(
            AssertionError, f"Method type is not valid. Provided: {invalid_method}"
        ):
            self.post_processor.create_postprocessor(invalid_method)

    def test_valid_create_post_processor(self):

        post_processor = self.post_processor.create_postprocessor("adam")
        # error message in case if test case got failed
        message = "given object is not instance of Adam class."
        self.assertIsInstance(post_processor, PostProcessorAdam, message)
