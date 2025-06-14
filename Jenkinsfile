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


        stage('Check Outdated Packages') {
            steps {
                script {
                    def result = sh(script: '''
                        pip list --outdated --format=json > outdated_packages.json
                        python3 -c "
                        import json
                        with open('outdated_packages.json') as f:
                        data = json.load(f)
                        print(len(data))"''', returnStdout: true).trim()

                    if (result != "0") {
                        echo "⚠️ Found ${result} outdated packages."
                        currentBuild.result = 'UNSTABLE'  // mark as UNSTABLE
                    } else {
                        echo "✅ All packages are up to date."
                    }
                }
            }
        }


        stage('Test') {
            steps {
                echo "Running tests..."
                

                // Run test commands here
                bat 'pytest tests/'
                
                
                // Run windows command prompt , then call batch file
                // bat 'if not exist reports mkdir reports'
                // bat 'pytest --junitxml=reports\\results.xml'

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
            archiveArtifacts artifacts: 'outdated_packages.json', onlyIfSuccessful: false
            // junit 'reports/results.xml'
        }
        success {
            echo "Build succeeded!"
        }
        failure {
            echo "Build failed."
        }
    }
}