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

// #########################
// Command lines prompt :

// Windows 	: bat
// Runs commands in cmd.exe

// Unix 	: sh or bash (which is an enhanced version of 'sh')
// Shell [sh]
// Default shell is /bin/sh
//
// Bash shell [bash]
// sh with #!/bin/bash or bash -c	For bash-specific syntax
//
// Quotes ...
// 🔸 When to use single quotes '...' in sh
// Use it when:
// You're running simple one-liner commands.
// You don't need to inject any Groovy/Environment variables.
// You want to avoid issues with unintended variable expansion.
// 
// 💡 Best Practice
// Use sh """ ... """ when injecting Jenkins/Groovy variables. This resolved $ variables !
// Use sh ''' ... ''' when using only shell-side variables and to avoid escaping $.
// 
// | Syntax           | Interpolation | Multiline | Use Case                          |
// | ---------------- | ------------- | --------- | --------------------------------- |
// | `sh 'cmd'`       | ❌ No          | ❌ No      | Simple one-liner, no variables    |
// | `sh "cmd $VAR"`  | ✅ Yes         | ❌ No      | One-liner with variable expansion |
// | `sh ''' cmd '''` | ❌ No          | ✅ Yes     | Multiline, shell expands vars     |
// | `sh """ cmd """` | ✅ Yes         | ✅ Yes     | Multiline, Groovy expands vars    |
//
// | Syntax                                  | Description                                      |
// | --------------------------------------- | ------------------------------------------------ |
// | `sh 'command'`                          | Simple one-liner                                 |
// | `sh '''...'''`                          | Multiline, positional                            |
// | `sh script: '...'`                      | Explicit, flexible (used when combining options) |
// | `sh(script: '...', returnStatus: true)` | Best when handling exit codes manually           |
//
// | Context               | Where output appears                           |
// | --------------------- | ---------------------------------------------- |
// | `echo "..."` (Groovy) | Jenkins pipeline log, as `[Pipeline] echo ...` |
// | `echo "..."` in `sh`  | Shell step output in Jenkins console           |
//
// | Scenario                            | Use `&&`? | Notes                            |
// | ----------------------------------- | --------- | -------------------------------- |
// | Run next only if previous succeeded | Yes       | Ensures stop-on-failure behavior |
// | Run all commands regardless         | No        | All commands run, even if errors |
// | Using `set -e` in script            | Optional  | `set -e` stops on failure anyway |
//
// source : ChatGPT
// #########################

pipeline {
    
    agent {
        label 'Jenkins_Node_Python_AudioLogger'
    }
    
    
    environment {
        
        REPO = 'MirkoThulke/Python_AudioLogger'
		OUTPUT_DIR = "${WORKSPACE}"
		PATH = "/usr/local/bin:${env.PATH}"
		
        // Virtual environment
        CONDA_BASE = "${HOME}/miniconda3"
        CONDA_ENV = "wxenv"
		
		// use bash by default. This line is not reliable.
		SHELL = '/bin/bash'
    }
	

	
	options {
		// Define how many builds shall be kept in the history
		buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '7'))
	}

	
    stages {
        // put individual stages here ....
        // each stage represents a jenkins pipeline stage
         
        stage('Set IS_UNIX variable') {
            steps {
                    script {
                        if (isUnix()) {
                            env.IS_UNIX = "true"
                        } else {
                            env.IS_UNIX = "false"
                        }
                    }
                    
                    // Use double quotes and Groovy interpolation to pass the env var value into the shell
                    echo "IS_UNIX is ${env.IS_UNIX}"
                }
        }

        
    
        // Check resource on Cloud instance
        stage('Check System Resources') {
                    steps {
                        script {
                            if (isUnix()) {
							    sh '''
							        free -h
							        sudo du -h / | sort -rh | head -n 20
							        df -h
							        lsblk
							    '''
                            }
                        } 
        
                    }
        }           
		
		
		// Clean workingspace and temporary variables
		stage('Clean System') {
					steps {
						script {
							
								cleanWs() // Deletes workspace after build

								if (isUnix()) {
									sh '''
										sudo apt clean
										sudo apt autoclean
										sudo journalctl --vacuum-time=2d
										sudo find /tmp -mindepth 1 -delete
										sudo find /var/tmp -mindepth 1 -delete
									'''
								}
						}
		
					}
		}
                        
		
        stage('Checkout Github') {
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
					/* .trim() is used here to remove any leading and trailing whitespace, 
					 * including newline characters, from the output of the sh command.*/
						
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
									sh script: '''
										curl --fail -H "Authorization: token $GITHUB_TOKEN" \
										-H "Accept: application/vnd.github.v3+json" \
										-X POST https://api.github.com/repos/${REPO}/statuses/${COMMIT_SHA} \
										-d '{"state": "pending", "context": "jenkins/build", "description": "Build started"}'
										''', env: [REPO: env.REPO, COMMIT_SHA: env.COMMIT_SHA]
                                } else {
									bat script: """
										curl -H "Authorization: token %GITHUB_TOKEN%" ^
										-H "Accept: application/vnd.github.v3+json" ^
										-X POST https://api.github.com/repos/${env.REPO}/statuses/${env.COMMIT_SHA} ^
										-d "{\\"state\\": \\"pending\\", \\"context\\": \\"jenkins/build\\", \\"description\\": \\"Build started\\"}"
										"""
                                }
								/* ${REPO} and ${COMMIT_SHA} are shell variables, not Groovy variables.
								* The shell itself interprets those variables during execution. */
                            }
                    }
            }
        }

		stage('Update Linux') {
			steps {
					echo "Checking if Linux packages must be updated ..."
					script {
					
						if (isUnix()) {
							sh '''
								sudo apt-get update
							'''
						} 
		
					}
			}
		}
		
		stage('Update Python') {
			steps {
					echo "Checking if python packages must be updated ..."
					script {
                        

						if (isUnix()) {
							echo "Activating Conda environment: ${env.CONDA_ENV}"
							sh(script: 
								"""
								set -e
								source ${env.CONDA_BASE}/etc/profile.d/conda.sh
								conda activate ${env.CONDA_ENV}
								echo 'Installing packages...'
								pip install --upgrade pip
								pip install --upgrade -r ${env.WORKSPACE}/requirements_linux.txt
								"""
							 	, shell: "/bin/bash")
								// only 'bash' supports 'source'. Force bash mode ! 
						} else {
							bat '''
								REM remove of old cache files first
								pip cache purge
								pip install --upgrade pip
								pip install --upgrade -r %WORKSPACE%\\requirements_windows.txt
								REM remove of old cache files first
								pip cache purge
							'''
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
						echo "Activating Conda environment: ${env.CONDA_ENV}"
						error_flag = sh(
							script: '''
								set -e
								source $CONDA_BASE/etc/profile.d/conda.sh
								conda activate $CONDA_ENV
								pytest --junitxml=report_integration_test_config.xml --capture=tee-sys tests/integration_tests/test_pythonConfig.py
							''',
							shell: '/bin/bash',
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
						echo "Activating Conda environment: ${env.CONDA_ENV}"
						error_flag = sh(
							script: '''
								set -e
						        source $CONDA_BASE/etc/profile.d/conda.sh
						        conda activate $CONDA_ENV
						        pytest --junitxml=report_integration_test_functional.xml --capture=tee-sys tests/integration_tests/test_integration_audioProcessing.py
							''',
							shell: '/bin/bash',
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
						echo "Activating Conda environment: ${env.CONDA_ENV}"
						error_flag_lowpass = sh(
							script: '''
								set -e
									source $CONDA_BASE/etc/profile.d/conda.sh
									conda activate $CONDA_ENV
									pytest --junitxml=report_lowpass.xml --capture=tee-sys tests/unit_tests/test_unit_lowpass.py
							''',
							shell: '/bin/bash',
							returnStatus: true
						)
						error_flag_aweighted = sh(
							script: '''
								set -e
								source $CONDA_BASE/etc/profile.d/conda.sh
								conda activate $CONDA_ENV
								pytest --junitxml=report_aweighted.xml --capture=tee-sys tests/unit_tests/test_unit_aweighted.py
							''',
							shell: '/bin/bash',
							returnStatus: true
							)
							
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
				
                script { 
                    
                    
					def error_flag = 0
					
					if (isUnix()) {
						echo "Activating Conda environment: ${env.CONDA_ENV}"
						error_flag = sh(
							script: '''
								set -e
								source $CONDA_BASE/etc/profile.d/conda.sh
								conda activate $CONDA_ENV
								pytest --junitxml=report_smoke_test.xml --capture=tee-sys tests/smoke_tests/test_smoke_audioInput.py
							''',
							shell: '/bin/bash',
							returnStatus: true
						)

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
			script {
				
					def reports = [
						'report_integration_test_config.xml',
						'report_integration_test_functional.xml',
						'report_lowpass.xml',
						'report_aweighted.xml',
						'report_smoke_test.xml'
					]
				
					// keeps the XML as a downloadable build artifact.
					// Publish JUnit test results
					reports.each { report ->
							junit testResults: report, allowEmptyResults: true
							archiveArtifacts artifacts: report, fingerprint: true
					}
				
					
				
                echo "Pipeline finished."
				
			}
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