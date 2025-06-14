pipeline 
{

    agent 
    { 
        label 'Jenkins_Node_Python_AudioLogger' 
    }

    environment 
    {
        // Define environment variables if needed
        EXAMPLE_VAR = "Hello, Jenkins!"
        result = 1
    }


    stages 
    {
        
        
        stage('Checkout') 
        {
            steps 
            {
                // Pulls code from your GitHub repository
                checkout scm
            }
        }


        stage('Check Python config') 
        {
            steps 
                {
                    echo "Running integration tests..."
                
                    // Run test commands here
                    // to do : 
                    //  Check Python version
                    // Check pip versions are uptodate
                    bat 'pytest tests/config_test/test_python_config.py'
                    
                    script 
                    {
                        if (result != "0") 
                        {
                            echo "⚠️ Found ${result} outdated packages."
                            currentBuild.result = 'UNSTABLE'  // mark as UNSTABLE
                        } 
                        else 
                        {
                            echo "✅ All packages are up to date."
                        }           
                    }              
                }
        }


        stage('Integration Tests') 
        {
            steps 
            {
                echo "Running integration tests..."
                
                // Run test commands here
                // To Do :  Integration test
                // Run on complete Python Programming
                // process audio, save audio
                // Inject perfect sinus 
                // save perfect sinus 
                // A-weighted : at 1kHz, no filtering EXPECTED
                // Low Pass Filtered : ar 100 Hz : No filtering expected
                // Check via signal convuluation in time domain
                
                // also : check that wave files are stored due to deteted noise
                
                bat 'pytest tests/integration_tests/test_integration_01.py'
                
            }
        }
        
        
        stage('Unit Tests') 
        {
            steps 
            {
                echo "Running unit tests..."
                
                // Run test commands here
                // To Do :  Integration test
                // Run individual functions
                // LowPass Filtering for example
                // A- WEIGHTING
                
                
            }
        }
        
        
        stage('Smoke Test') 
        {
            steps 
            {
                echo "Running smoke test..."
                
                // Run test commands here
                // To Do :  Integration test
                //    - Detect a USB microphone,
                //    - Capture audio, and
                //    - Store it as a WAV file
                
                
            }
        }  
        
        
        stage('Deploy') 
        {
            steps 
            {
                echo "Deploying the application..."
                // Deploy logic goes here (e.g., copy files, Docker deploy, etc.)
            }
        }
        
    }


    post 
    {
        always 
        {
            echo "Pipeline finished."
        }
        success 
        {
            echo "Build succeeded!"
        }
        failure 
        {
            echo "Build failed."
        }
    }
    
    
}