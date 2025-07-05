pipeline {
    agent {
        label 'Jenkins_Node_Python_AudioLogger'
    }

    environment {
        EXAMPLE_VAR = "Hello, Jenkins!"
    }

    stages {
        // put individual stages here ....
        // each stage represents a jenkins pipeline stage
          
        stage('Validate Python Configuration') {
            steps {
                script {
                    // Freeze installed packages
                    bat 'pip freeze > requirements_new.txt'

                    // Compare with requirements.txt
                    def compareResult = bat(
                        script: 'fc /B requirements.txt requirements_new.txt > nul',
                        returnStatus: true
                    )
                    if (compareResult != 0) {
                        echo 'Installed packages differ from requirements.txt!'
                        bat 'fc requirements.txt requirements_new.txt'
                        currentBuild.result = 'UNSTABLE'
                    }

                    // Check for broken dependencies
                    def checkDeps = bat(
                        script: 'pip check',
                        returnStatus: true
                    )
                    if (checkDeps != 0) {
                        echo 'Broken dependencies detected!'
                        currentBuild.result = 'UNSTABLE'
                    }

                    // List outdated packages
                    bat 'pip list --outdated > outdated-packages.txt'
                    bat 'echo Outdated packages listed in outdated-packages.txt'

                    // Check if outdated-packages.txt is not empty and mark build UNSTABLE
                    def outdated = readFile('outdated-packages.txt').trim()
                    if (outdated) {
                        echo 'Outdated packages found. Marking build as UNSTABLE.'
                        currentBuild.result = 'UNSTABLE'
                    } else {
                        echo 'No outdated packages found.'
                    }
                }
            }
        }
        
        

        stage('Checkout') {
            steps {
                checkout scm
            }
        }


        stage('Integration Test - Python Config') {
            steps {
                echo "🔍 Running Python config integration test..."

                script {
                    def error_flag = bat(script: 'pytest --junitxml=report.xml tests/integration_tests/test_pythonConfig.py', returnStatus: true)

                    if (error_flag != 0) {
                        echo "⚠️ Integration test failed with code ${error_flag}."
                        currentBuild.result = 'UNSTABLE'
                    } else {
                        echo "✅ Integration test passed. All good."
                    }
                }
            }
        }



        stage('Integration Tests - Functional') 
        {
            steps
            {
                echo "🔧 Running functional integration tests..."

                script 
                {
                    def error_flag = bat(script: 'pytest --junitxml=report.xml tests/integration_tests/test_integration_audioProcessing.py', returnStatus: true)

                }
            }
        }



        stage('Unit Test') {
            steps {
                echo "🧪 Running unit tests..."

                script {
                    def error_flag = bat(script: 'pytest --junitxml=report.xml tests/unit_tests/test_unit_lowpass.py', returnStatus: true)

                }
            } 
        }




        stage('Smoke Test') {
            steps {
                echo "🧪 Running smoke tests..."

                script {
                    
                    def error_flag = bat(script: 'pytest --junitxml=report.xml tests/smoke_tests/test_smoke_audioInput.py', returnStatus: true)

                }
            }
        }



        stage('Deploy') {
            steps {
                echo "🚀 Deploying the application..."
                // Your deploy logic here
            }
        }
    }



    post {
        
        always {
            junit 'report.xml' // publish results
            echo "🏁 Pipeline finished."
        }
        success {
            echo "✅ Build succeeded!"
        }
        failure {
            echo "❌ Build failed."
        }
    }
}