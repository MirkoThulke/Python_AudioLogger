// -----------------------------------------------------------------
// Author: MIRKO THULKE 
// Copyright (c) 2025, MIRKO THULKE
// All rights reserved.
//
// Date: 2025, VERSAILLES, FRANCE
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING
// FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// -----------------------------------------------------------------------------



// pytest options :
// pytest --junitxml=report.xml  : creates a report, which includes error messages
// pytest--capture=tee-sys : also adds print statements to the report
// pytest : by default the jenkins pipeline will stop and fail if test functions fail
// pytest : currentBuild.result = 'UNSTABLE' -> this attribute allows , to let the build pass, but mark it as partly failed.

// #########################
// Pytest calls all functions starting with "test_" automatically
// hence, functions to be called by pytest MUST start with "test_"

// Unit test howto :
// https://youtu.be/6tNS--WetLI?feature=shared




pipeline {
    
    agent {
        label 'Jenkins_Node_Python_AudioLogger'
    }
    
    
    environment {
        REPO = 'MirkoThulke/Python_AudioLogger'
    }


    stages {
        // put individual stages here ....
        // each stage represents a jenkins pipeline stage
          
        
        stage('Checkout') {
            steps {
                    checkout scm
            }
        }


        stage('Capture Git Commit SHA') {
            steps {
                script {
                    try {
                        if (isUnix()) {
                            env.COMMIT_SHA = sh(
                                script: 'git rev-parse HEAD',
                                returnStdout: true
                            ).trim()
                        } else {
                            env.COMMIT_SHA = bat(
                                script: '@echo off & git rev-parse HEAD',
                                returnStdout: true
                            ).trim()
                        }
                        echo "COMMIT_SHA is: ${env.COMMIT_SHA}"
                    } catch (e) {
                        error "Failed to retrieve Git commit SHA: ${e}"
                    }
                }
            }
        }
        
        
        stage('Notify GitHub - Pending') {
            steps {
                    script {
                        
                            withCredentials([string(credentialsId: 'mirko-github-api-token', variable: 'GITHUB_TOKEN')]) {
                                if (isUnix()) {
                                    sh """
                                        curl -H "Authorization: token ${GITHUB_TOKEN}" \\
                                        -H "Accept: application/vnd.github.v3+json" \\
                                        -X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} \\
                                        -d '{\"state\": \"pending\", \"context\": \"jenkins/build\", \"description\": \"Build started\"}'
                                        """
                                } else {
                                    bat """
                                        curl -H "Authorization: token %GITHUB_TOKEN%" ^
                                        -H "Accept: application/vnd.github.v3+json" ^
                                        -X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} ^
                                        -d "{\\"state\\": \\"pending\\", \\"context\\": \\"jenkins/build\\", \\"description\\": \\"Build started\\"}"
                                        """
                                }
                            }
                    }
            }
        }


        stage('Integration Test - Python Config') {
            steps {
                echo "Running Python config integration test..."
        
                script {
                    def error_flag = 0
					
                    if (isUnix()) {
                        error_flag = sh(
                            script: 'pytest --junitxml=report_integration_test_config.xml --capture=tee-sys tests/integration_tests/test_pythonConfig.py',
                            returnStatus: true
                        )
                    } else {
                        error_flag = bat(
                            script: 'pytest --junitxml=report_integration_test_config.xml --capture=tee-sys tests/integration_tests/test_pythonConfig.py',
                            returnStatus: true
                        )
                    }
        
                    if (error_flag != 0) {
                        echo "Integration test failed with code ${error_flag}."
                        currentBuild.result = 'UNSTABLE' // Or use 'FAILURE' if stricter
                    } else {
                        echo "Integration test passed. All good."
                    }
                }
            }
        }



        stage('Integration Tests - Functional') {
            steps {
                echo "Running functional integration tests..."
        
                script {
                    def error_flag = 0
					
                    if (isUnix()) {
                        error_flag = sh(
                            script: 'pytest --junitxml=report_integration_test_functional.xml --capture=tee-sys tests/integration_tests/test_integration_audioProcessing.py',
                            returnStatus: true
                        )
                    } else {
                        error_flag = bat(
                            script: 'pytest --junitxml=report_integration_test_functional.xml --capture=tee-sys tests/integration_tests/test_integration_audioProcessing.py',
                            returnStatus: true
                        )
                    }
        
                    if (error_flag != 0) {
                        error "Functional integration test failed with exit code ${error_flag}."
                    } else {
                        echo "Functional test passed. All good."
                    }
                }
            }
        }


        stage('Unit Test') {
            steps {
                echo "Running unit tests..."

                script {
					def error_flag_lowpass = 0
					def error_flag_aweighted = 0
					
					if (isUnix()) {
						error_flag_lowpass = sh(script: 'pytest --junitxml=report_lowpass.xml --capture=tee-sys tests/unit_tests/test_unit_lowpass.py', returnStatus: true)
						error_flag_aweighted = sh(script: 'pytest --junitxml=report_aweighted.xml --capture=tee-sys tests/unit_tests/test_unit_aweighted.py', returnStatus: true)
					} else {
						error_flag_lowpass = bat(script: 'pytest --junitxml=report_lowpass.xml --capture=tee-sys tests/unit_tests/test_unit_lowpass.py', returnStatus: true)
						error_flag_aweighted = bat(script: 'pytest --junitxml=report_aweighted.xml --capture=tee-sys tests/unit_tests/test_unit_aweighted.py', returnStatus: true)
					}
					
					if (error_flag_lowpass != 0 || error_flag_aweighted != 0) {
						error "Unit tests failed!"
					}
					
				}
            } 
        }


        stage('Smoke Test') {
            steps {
                echo "Running smoke tests..."
				
				def error_flag = 0
				
                script { 
                    
					if (isUnix()) {
						error_flag = sh(script: 'pytest --junitxml=report_smoke_test.xml --capture=tee-sys tests/smoke_tests/test_smoke_audioInput.py', returnStatus: true)
					} else {
						error_flag = bat(script: 'pytest --junitxml=report_smoke_test.xml --capture=tee-sys tests/smoke_tests/test_smoke_audioInput.py', returnStatus: true)
					}
				}
            }
        }



        stage('Deploy') {
            steps {
                echo "Deploying the application..."
                // Your deploy logic here
            }
        }
    }



    post {

        always {
                junit 'report_integration_test_config.xml'  // publish results
				junit 'report_integration_test_functional.xml'  // publish results
				junit 'report_lowpass.xml'  // publish results
				junit 'report_aweighted.xml'  // publish results
				junit 'report_smoke_test.xml'  // publish results
                echo "Pipeline finished."
        }
    
		
		success {
			script {
				withCredentials([string(credentialsId: 'mirko-github-api-token', variable: 'GITHUB_TOKEN')]) {
					if (isUnix()) {
						sh """
                    curl -H "Authorization: token \$GITHUB_TOKEN" \\
                         -H "Accept: application/vnd.github.v3+json" \\
                         -X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} \\
                         -d '{\\"state\\": \\"success\\", \\"context\\": \\"jenkins/build\\", \\"description\\": \\"Build succeeded\\"}'
                """
					} else {
						bat """
                    curl -H "Authorization: token %GITHUB_TOKEN%" ^
                         -H "Accept: application/vnd.github.v3+json" ^
                         -X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} ^
                         -d "{\\"state\\": \\"success\\", \\"context\\": \\"jenkins/build\\", \\"description\\": \\"Build succeeded\\"}"
                """
					}
				}
			}
		}
		
		failure {
			script {
				withCredentials([string(credentialsId: 'mirko-github-api-token', variable: 'GITHUB_TOKEN')]) {
					if (isUnix()) {
						sh """
                    curl -H "Authorization: token \$GITHUB_TOKEN" \\
                         -H "Accept: application/vnd.github.v3+json" \\
                         -X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} \\
                         -d '{\\"state\\": \\"failure\\", \\"context\\": \\"jenkins/build\\", \\"description\\": \\"Build failed\\"}'
                """
					} else {
						bat """
                    curl -H "Authorization: token %GITHUB_TOKEN%" ^
                         -H "Accept: application/vnd.github.v3+json" ^
                         -X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} ^
                         -d "{\\"state\\": \\"failure\\", \\"context\\": \\"jenkins/build\\", \\"description\\": \\"Build failed\\"}"
                """
					}
				}
			}
		}

		unstable {
			script {
				withCredentials([string(credentialsId: 'mirko-github-api-token', variable: 'GITHUB_TOKEN')]) {
					if (isUnix()) {
						sh """
                    curl -H "Authorization: token \$GITHUB_TOKEN" \\
                         -H "Accept: application/vnd.github.v3+json" \\
                         -X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} \\
                         -d '{\\"state\\": \\"pending\\", \\"context\\": \\"jenkins/build\\", \\"description\\": \\"Build unstable\\"}'
							"""
					} else {
						bat """
                    curl -H "Authorization: token %GITHUB_TOKEN%" ^
                         -H "Accept: application/vnd.github.v3+json" ^
                         -X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} ^
                         -d "{\\"state\\": \\"pending\\", \\"context\\": \\"jenkins/build\\", \\"description\\": \\"Build unstable\\"}"
						"""
					}
				}
			}
		}
		
    } // post			
} //pipeline