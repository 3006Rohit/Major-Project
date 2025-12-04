"""
DEPLOYMENT & TESTING GUIDE
Complete instructions for deploying Stock Price Prediction System
"""

# ============================================================================
# 1. SYSTEM REQUIREMENTS
# ============================================================================

REQUIREMENTS = """
System Requirements:
- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- 2GB disk space for models
- Internet connection for data fetching

OS Support:
✓ Windows 10/11
✓ macOS (Intel & Apple Silicon)
✓ Linux (Ubuntu, Debian, etc.)
✓ Docker (any OS)

Dependencies installed via: pip install -r requirements.txt
"""

# ============================================================================
# 2. LOCAL DEPLOYMENT
# ============================================================================

LOCAL_SETUP = """
STEP 1: Initial Setup
=====================================
1. Clone repository:
   git clone <repo-url>
   cd stock_price_prediction

2. Create virtual environment:
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\\Scripts\\activate     # Windows

3. Install dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt

4. Verify installation:
   python -c "import tensorflow; print('TensorFlow OK')"
   python -c "import xgboost; print('XGBoost OK')"
   python -c "import yfinance; print('yfinance OK')"

STEP 2: Run Main Pipeline
=====================================
python stock_prediction.py

Output:
- model_comparison.png: Visual comparison
- technical_indicators.png: Technical plots
- model_results.csv: Detailed metrics

STEP 3: Run Streamlit App (Recommended)
=====================================
streamlit run app_streamlit.py

Access: http://localhost:8501
Features:
- Real-time stock data
- Interactive charts
- Model predictions
- Performance comparison

STEP 4: Generate HTML Dashboard
=====================================
python dashboard_generator.py

Output: dashboard.html (open in browser)
"""

# ============================================================================
# 3. DOCKER DEPLOYMENT
# ============================================================================

DOCKER_DEPLOYMENT = """
STEP 1: Build Docker Image
=====================================
docker build -t stock-predictor:latest .

Verify:
docker images | grep stock-predictor

STEP 2: Run Container
=====================================
docker run -p 8501:8501 stock-predictor:latest

Access: http://localhost:8501

STEP 3: Run with Volume Mounting
=====================================
docker run -p 8501:8501 \\
  -v $(pwd)/data:/app/data \\
  -v $(pwd)/models:/app/models \\
  -v $(pwd)/results:/app/results \\
  stock-predictor:latest

STEP 4: Docker Compose
=====================================
version: '3.8'
services:
  stock-predictor:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
    environment:
      - PYTHONUNBUFFERED=1

Run: docker-compose up
"""

# ============================================================================
# 4. AWS DEPLOYMENT
# ============================================================================

AWS_DEPLOYMENT = """
OPTION 1: AWS ECS (Elastic Container Service)
=====================================
1. Create ECR Repository:
   aws ecr create-repository --repository-name stock-predictor

2. Get login token:
   aws ecr get-login-password --region us-east-1 | \\
   docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

3. Tag and push image:
   docker tag stock-predictor:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/stock-predictor:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/stock-predictor:latest

4. Create ECS task definition and service

OPTION 2: AWS Lambda
=====================================
1. Create Lambda function
2. Upload Docker image as container
3. Set memory: 3008 MB
4. Set timeout: 900 seconds
5. Add API Gateway trigger

OPTION 3: AWS EC2
=====================================
1. Launch EC2 instance (t3.large)
2. SSH into instance
3. Install Docker
4. Pull and run container
5. Attach Elastic IP
6. Configure security groups

Cost Estimate:
- ECS: $0.05/vCPU-hour + $0.01/GB-hour
- Lambda: $0.20 per 1M requests + compute time
- EC2: ~$30-50/month for t3.large
"""

# ============================================================================
# 5. HEROKU DEPLOYMENT
# ============================================================================

HEROKU_DEPLOYMENT = """
STEP 1: Create Heroku App
=====================================
heroku login
heroku create stock-price-predictor

STEP 2: Set Buildpack
=====================================
heroku buildpacks:set heroku/python

STEP 3: Configure Procfile
=====================================
Create Procfile:
web: streamlit run app_streamlit.py --server.port=$PORT --server.address=0.0.0.0

STEP 4: Set Stack
=====================================
heroku stack:set container

STEP 5: Deploy
=====================================
git push heroku main

STEP 6: View Logs
=====================================
heroku logs --tail

Access: https://stock-price-predictor.herokuapp.com

Note: Free tier has 30-min sleep limit. Use Eco Dyno ($7/mo) for production.
"""

# ============================================================================
# 6. TESTING
# ============================================================================

TESTING = """
Unit Tests
=====================================
pytest tests/ -v

Test Coverage:
pytest --cov=src tests/

Key Tests:
✓ Data fetching accuracy
✓ Feature engineering correctness
✓ Model training convergence
✓ Prediction generation
✓ Metrics calculation
✓ Visualization generation

Integration Tests
=====================================
python -m pytest tests/integration/ -v

Full Pipeline Test:
python tests/test_full_pipeline.py

Performance Tests
=====================================
python -m pytest tests/performance/ -v

Checks:
- Model training time < 5 minutes
- Prediction latency < 100ms
- Memory usage < 2GB
"""

# ============================================================================
# 7. PRODUCTION CHECKLIST
# ============================================================================

PRODUCTION_CHECKLIST = """
Pre-Deployment Checklist
=====================================
[ ] All tests passing (pytest)
[ ] Code coverage > 80%
[ ] No hardcoded credentials
[ ] Environment variables configured
[ ] Logging configured properly
[ ] Error handling implemented
[ ] Rate limiting added (if API)
[ ] Database backups setup
[ ] Monitoring alerts configured
[ ] Documentation updated
[ ] Security scan completed (bandit)

Deployment Checklist
=====================================
[ ] Database migrations run
[ ] Model files uploaded
[ ] Configuration files synced
[ ] Environment variables set
[ ] SSL certificates valid
[ ] DNS records updated
[ ] Monitoring dashboards created
[ ] Backup and recovery tested
[ ] Load testing completed
[ ] Performance benchmarks met

Post-Deployment
=====================================
[ ] Health checks passing
[ ] Logs being collected
[ ] Alerts being triggered
[ ] User acceptance testing
[ ] Analytics tracking working
[ ] Error tracking active
[ ] Performance monitoring live
"""

# ============================================================================
# 8. MONITORING & LOGGING
# ============================================================================

MONITORING = """
Monitoring Setup
=====================================
Key Metrics to Monitor:
- Model prediction accuracy
- API response time
- Error rate
- CPU/Memory usage
- Data freshness

Tools:
- Prometheus: Metrics collection
- Grafana: Visualization
- ELK Stack: Logging
- Datadog: Full observability

Configuration Example:
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

Health Checks
=====================================
/health - Basic health check
/metrics - Prometheus metrics
/status - Detailed status

Example Response:
{
  "status": "healthy",
  "uptime": 3600,
  "models_loaded": 10,
  "last_prediction": "2024-01-15 10:30:00",
  "memory_usage_mb": 1024
}
"""

# ============================================================================
# 9. PERFORMANCE OPTIMIZATION
# ============================================================================

OPTIMIZATION = """
Model Optimization
=====================================
1. Model Quantization:
   - Float32 → Float16 (50% size reduction)
   - int8 quantization for inference
   
2. Model Pruning:
   - Remove 10-20% less important weights
   - Reduces size by 20-30%

3. Model Distillation:
   - Train smaller model from larger model
   - 90% accuracy with 50% size

4. ONNX Conversion:
   - Convert models to ONNX format
   - 2-3x faster inference

Code Example:
import onnx
from tensorflow.keras import models

model = models.load_model('model.h5')
onnx_model = keras2onnx.convert_model(model)
onnx.save_model(onnx_model, 'model.onnx')

Inference Optimization:
- Batch predictions (10-50 samples)
- Use GPU acceleration (CUDA)
- Cache model in memory
- Async processing

Deployment Optimization:
- Load balance across multiple servers
- Redis caching for results
- CDN for static assets
- Database indexing
"""

# ============================================================================
# 10. TROUBLESHOOTING
# ============================================================================

TROUBLESHOOTING = """
Common Issues & Solutions
=====================================

Issue: Model Training Too Slow
Solution:
- Reduce lookback_days
- Use smaller dataset for testing
- Enable GPU: tf.config.list_physical_devices('GPU')
- Use XGBoost (faster than LSTM)

Issue: Out of Memory
Solution:
- Reduce batch_size
- Reduce sequence length
- Use float16 instead of float32
- Process data in chunks

Issue: Poor Predictions
Solution:
- Check data quality
- Add more indicators
- Tune hyperparameters
- Use ensemble method
- Increase training data

Issue: Slow API Response
Solution:
- Use model quantization
- Implement caching
- Batch predictions
- Use GPU acceleration
- Load balancing

Issue: yfinance No Data
Solution:
- Check internet connection
- Verify stock symbol correct
- Use VPN if blocked
- Try alpha_vantage API alternative
- Check market hours

Debugging
=====================================
Enable debug mode:
import logging
logging.basicConfig(level=logging.DEBUG)

Add print statements strategically:
print(f"Data shape: {data.shape}")
print(f"Features: {feature_cols}")
print(f"Model prediction: {prediction}")

Use debugger:
import pdb; pdb.set_trace()

Check logs:
tail -f app.log
"""

# ============================================================================
# 11. BACKUP & DISASTER RECOVERY
# ============================================================================

DISASTER_RECOVERY = """
Backup Strategy
=====================================
1. Database Backups:
   - Daily automated backups
   - 30-day retention
   - Test restore quarterly

2. Model Backups:
   - Version control (Git)
   - S3/GCS cloud storage
   - Local copies

3. Configuration Backups:
   - Environment files
   - Model configs
   - Feature definitions

Disaster Recovery Plan
=====================================
RTO (Recovery Time Objective): < 1 hour
RPO (Recovery Point Objective): < 15 minutes

Steps:
1. Identify failure
2. Activate backup system
3. Verify data integrity
4. Restore to new instance
5. Run health checks
6. Switch traffic
7. Monitor closely

Test recovery monthly!
"""

# ============================================================================
# 12. SECURITY
# ============================================================================

SECURITY = """
Security Measures
=====================================
1. Code Security:
   - No hardcoded passwords
   - Input validation
   - SQL injection prevention
   - XSS protection

2. API Security:
   - Authentication tokens
   - Rate limiting
   - CORS configuration
   - HTTPS only

3. Data Security:
   - Encrypt data at rest
   - Encrypt data in transit
   - Secure API keys
   - PII handling compliance

4. Deployment Security:
   - Use secrets manager
   - Least privilege access
   - Network isolation
   - Regular security scans

Security Checklist:
[ ] No API keys in code
[ ] HTTPS enabled
[ ] Input validation added
[ ] Rate limiting implemented
[ ] Authentication required
[ ] Data encrypted
[ ] Regular backups
[ ] Security headers set
[ ] Dependencies updated
[ ] Penetration testing done

Tools:
- bandit: Find security issues
- safety: Check dependencies
- snyk: Vulnerability scanning
"""

if __name__ == "__main__":
    print("STOCK PRICE PREDICTION - DEPLOYMENT GUIDE")
    print("=" * 60)
    print("\nSelect deployment option:")
    print("1. Local Setup")
    print("2. Docker Deployment")
    print("3. AWS Deployment")
    print("4. Heroku Deployment")
    print("5. View All Guides")
    
    option = input("\nEnter option (1-5): ")
    
    guides = {
        '1': LOCAL_SETUP,
        '2': DOCKER_DEPLOYMENT,
        '3': AWS_DEPLOYMENT,
        '4': HEROKU_DEPLOYMENT,
        '5': f"{LOCAL_SETUP}\n\n{DOCKER_DEPLOYMENT}\n\n{AWS_DEPLOYMENT}\n\n{HEROKU_DEPLOYMENT}"
    }
    
    if option in guides:
        print("\n" + guides[option])
    else:
        print("Invalid option. Exiting.")
