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


        stage('Check Python Packages') {
           steps {
               script {
                   catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
                       bat '''
                            pip list --outdated --format=json > outdated_packages.json
        
                            setlocal enabledelayedexpansion
                            set COUNT=0
                            for /f "usebackq" %%A in (`findstr /R /N "^" outdated_packages.json`) do 
                            (
                            set /a COUNT+=1
                            )
                            echo Total lines: !COUNT!
        
                            if %COUNT% GTR 2 (
                               echo WARNING: Outdated Python packages found.
                               type outdated_packages.json
                               exit /b 1
                            ) else (
                               echo All Python packages are up to date.
                               exit /b 0
                           )
                       '''
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