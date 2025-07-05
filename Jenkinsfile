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