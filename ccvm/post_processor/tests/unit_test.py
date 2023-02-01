import logging
import time
import unittest
import uuid
import xmlrunner

from ccvm.post_processor.tests.test_post_processor_Adam import TestPostProcessorAdam

# import config


def suite():
    # unit test suite
    unit_test_suite = unittest.TestSuite()

    ########################
    # PostProcessorAdam    #
    ########################
    unit_test_suite.addTest(TestPostProcessorAdam("test_postprocess_valid"))
    unit_test_suite.addTest(
        TestPostProcessorAdam("test_postprocess_invalid_c_parameter")
    )
    unit_test_suite.addTest(
        TestPostProcessorAdam("test_postprocess_invalid_qmat_parameter")
    )
    unit_test_suite.addTest(
        TestPostProcessorAdam("test_postprocess_invalid_c_vector_parameter")
    )
    unit_test_suite.addTest(
        TestPostProcessorAdam("test_postprocess_error_for_invalid_c_dimension")
    )
    unit_test_suite.addTest(
        TestPostProcessorAdam("test_postprocess_error_for_invalid_c_vector_shape")
    )
    return unit_test_suite


if __name__ == "__main__":
    runner = xmlrunner.XMLTestRunner(output="./")
    logger = logging.getLogger()
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"========== UnitTest Started, ID: {run_id} ==========")
    start_time = time.time()
    test_result = runner.run(suite())
    execution_time = round(time.time() - start_time, 2)
    test_status = "FAILED"
    if test_result.wasSuccessful():
        test_status = "OK"
    logger.info(
        f"========== Unit Test Finished, ID: {run_id}, Execution Time={execution_time}, {test_status} =========="
    )

    exit(not test_result.wasSuccessful())
