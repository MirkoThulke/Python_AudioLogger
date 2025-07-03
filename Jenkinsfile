pipeline {
    agent {
        label 'Jenkins_Node_Python_AudioLogger'
    }

    environment {
        EXAMPLE_VAR = "Hello, Jenkins!"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Integration Test - Python Config') {
            steps {
                echo "🔍 Running Python config integration test..."

                script {
                    def error_flag = bat(script: 'pytest tests/integration_tests/test_pythonConfig.py', returnStatus: true)

                    if (error_flag != 0) {
                        echo "⚠️ Integration test failed with code ${error_flag}."
                        currentBuild.result = 'UNSTABLE'
                    } else {
                        echo "✅ Integration test passed. All good."
                    }
                }
            }
        }

        stage('Integration Tests - Functional') {
            steps {
                echo "🔧 Running functional integration tests..."

                script {
                    def error_flag = bat(script: 'pytest tests/integration_tests/test_integration_audioProcessing.py', returnStatus: true)

                    if (error_flag != 0) {
                        error "❌ Functional tests failed with code ${error_flag}"
                    } else {
                        echo "✅ Functional tests passed. All good."
                    }
                }
            }
        }

        stage('Unit Test') {
            steps {
                echo "🧪 Running unit tests..."

                script {
                    def error_flag = bat(script: 'pytest tests/unit_tests/test_unit_lowpass.py', returnStatus: true)

                    if (error_flag != 0) {
                        error "❌ Unit tests failed with code ${error_flag}"
                    } else {
                        echo "✅ Unit tests passed. All good."
                    }
                }
            }
        }


        stage('Smoke Test') {
            steps {
                echo "🧪 Running smoke tests..."

                script {
                    def error_flag = bat(script: 'pytest tests/unit_tests/test_smoke_audioInput.py', returnStatus: true)

                    if (error_flag != 0) {
                        error "❌ Unit tests failed with code ${error_flag}"
                    } else {
                        echo "✅ Unit tests passed. All good."
                    }
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