# import logging
# import time
# import unittest
# import uuid
# import xmlrunner

# from ccvm_simulators.post_processor.tests.test_adam import TestPostProcessorAdam
# from ccvm_simulators.post_processor.tests.test_bfgs import TestPostProcessorBFGS
# from ccvm_simulators.post_processor.tests.test_asgd import TestPostProcessorASGD
# from ccvm_simulators.post_processor.tests.test_lbfgs import TestPostProcessorLBFGS
# from ccvm_simulators.post_processor.tests.test_trust_constr import (
#     TestPostProcessorTrustConstr,
# )
# from ccvm_simulators.post_processor.tests.test_factory import (
#     TestPostProcessorFactory,
# )
# from ccvm_simulators.post_processor.tests.test_box_qp_model import TestBoxQPModel

# # import config


# def suite():
#     # unit test suite
#     unit_test_suite = unittest.TestSuite()

#     #ADAM
#     unit_test_suite.addTest(TestPostProcessorAdam("test_postprocess_valid"))
#     unit_test_suite.addTest(
#         TestPostProcessorAdam("test_postprocess_invalid_c_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorAdam("test_postprocess_invalid_qmat_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorAdam("test_postprocess_invalid_c_vector_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorAdam("test_postprocess_error_for_invalid_c_dimension")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorAdam("test_postprocess_error_for_invalid_c_vector_shape")
#     )

#     # ASGD
#     unit_test_suite.addTest(TestPostProcessorASGD("test_postprocess_valid"))
#     unit_test_suite.addTest(
#         TestPostProcessorASGD("test_postprocess_invalid_c_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorASGD("test_postprocess_invalid_qmat_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorASGD("test_postprocess_invalid_c_vector_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorASGD("test_postprocess_error_for_invalid_c_dimension")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorASGD("test_postprocess_error_for_invalid_c_vector_shape")
#     )

#     # BFGS
#     unit_test_suite.addTest(TestPostProcessorBFGS("test_postprocess_valid"))
#     unit_test_suite.addTest(
#         TestPostProcessorBFGS("test_postprocess_invalid_c_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorBFGS("test_postprocess_invalid_qmat_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorBFGS("test_postprocess_invalid_c_vector_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorBFGS("test_postprocess_error_for_invalid_c_dimension")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorBFGS("test_postprocess_error_for_invalid_c_vector_shape")
#     )

#     # LBFGS
#     unit_test_suite.addTest(TestPostProcessorLBFGS("test_postprocess_valid"))
#     unit_test_suite.addTest(
#         TestPostProcessorLBFGS("test_postprocess_invalid_c_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorLBFGS("test_postprocess_invalid_qmat_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorLBFGS("test_postprocess_invalid_c_vector_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorLBFGS("test_postprocess_error_for_invalid_c_dimension")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorLBFGS("test_postprocess_error_for_invalid_c_vector_shape")
#     )

#     # TrustConstr
#     unit_test_suite.addTest(TestPostProcessorTrustConstr("test_postprocess_valid"))
#     unit_test_suite.addTest(
#         TestPostProcessorTrustConstr("test_postprocess_invalid_c_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorTrustConstr("test_postprocess_invalid_qmat_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorTrustConstr("test_postprocess_invalid_c_vector_parameter")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorTrustConstr("test_postprocess_error_for_invalid_c_dimension")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorTrustConstr(
#             "test_postprocess_error_for_invalid_c_vector_shape"
#         )
#     )

#     # ProcessFactory
#     unit_test_suite.addTest(
#         TestPostProcessorFactory("test_valid_create_post_processor")
#     )
#     unit_test_suite.addTest(
#         TestPostProcessorFactory("test_invalid_create_post_processor")
#     )

#     # BOXQP
#     unit_test_suite.addTest(TestBoxQPModel("test_invalid_boxqp"))
#     unit_test_suite.addTest(TestBoxQPModel("test_valid_boxqp_adam"))

#     return unit_test_suite


# if __name__ == "__main__":
#     runner = xmlrunner.XMLTestRunner(output="./test_output")
#     logger = logging.getLogger()
#     run_id = str(uuid.uuid4())[:8]
#     logger.info(f"========== UnitTest Started, ID: {run_id} ==========")
#     start_time = time.time()
#     test_result = runner.run(suite())
#     execution_time = round(time.time() - start_time, 2)
#     test_status = "FAILED"
#     if test_result.wasSuccessful():
#         test_status = "OK"
#     logger.info(
#         f"========== Unit Test Finished, ID: {run_id}, Execution Time={execution_time}, {test_status} =========="
#     )

#     exit(not test_result.wasSuccessful())
