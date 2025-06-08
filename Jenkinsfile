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

        stage('Check Python Setup') {
            steps {
                
                    // script {
                    //    githubNotify context: 'build', status: 'PENDING', description: 'Build is starting...'
                    // } 
                
                    // Add your build commands here
                    // For example: mvn clean install, npm install, etc.
                    
                    echo "Check Python for outdated packages..."
                    bat 'pip list --outdated > outdated_packages.txt'
                    
                    //script {
                    //    githubNotify context: 'build', status: 'SUCCESS', description: 'Build passed!'
                    //}
                
            }
        }

        stage('Test') {
            steps {
                echo "Running tests..."
                

                // Run test commands here
                bat 'pytest tests/'
                
                
                // Run windows command prompt , then call batch file
                bat '''
                    if not exist reports mkdir reports
                    pytest --junitxml=reports\\results.xml
                    dir reports
                '''
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
            junit 'reports/results.xml'
        }
        success {
            echo "Build succeeded!"
        }
        failure {
            echo "Build failed."
        }
    }
}
