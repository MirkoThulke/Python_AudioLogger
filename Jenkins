pipeline {
    agent any  // Runs on any available Jenkins agent

    environment {
        // Define environment variables if needed
        EXAMPLE_VAR = "Hello, Jenkins!"
    }

    stages {
        stage('Checkout') {
            steps {
                // Pulls code from your GitHub repository
                checkout scm
            }
        }

        stage('Build') {
            steps {
                echo "Building the project..."
                // Add your build commands here
                // For example: mvn clean install, npm install, etc.
            }
        }

        stage('Test') {
            steps {
                echo "Running tests..."
                // Run test commands here
            }
        }

        stage('Deploy') {
            steps {
                echo "Deploying the application..."
                // Deploy logic goes here (e.g., copy files, Docker deploy, etc.)
            }
        }
    }

    post {
        always {
            echo "Pipeline finished."
        }
        success {
            echo "Build succeeded!"
        }
        failure {
            echo "Build failed."
        }
    }
}
