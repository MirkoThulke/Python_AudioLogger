pipeline {

    // define Jenkins agent node which is specified for this jenkins job (best practice)
    agent { label 'Jenkins_Node_Python_AudioLogger' }
    // agent any  // Runs on any available Jenkins agent

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
                
                    script {
                        githubNotify context: 'build', status: 'PENDING', description: 'Build is starting...'
                    }
                
                    echo "Building the project..."
                    // Add your build commands here
                    // For example: mvn clean install, npm install, etc.
                    sleep 2 // simulate build
                    
                    script {
                        githubNotify context: 'build', status: 'SUCCESS', description: 'Build passed!'
                    }
                
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
