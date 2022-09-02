FROM public.ecr.aws/lambda/python:3.9

COPY classifier.py ${LAMBDA_TASK_ROOT}
COPY test_classifier.py ${LAMBDA_TASK_ROOT}
COPY features ${LAMBDA_TASK_ROOT}/features
COPY features/features_mean.npy ${LAMBDA_TASK_ROOT}/features
COPY features/features_mean_freq.npy ${LAMBDA_TASK_ROOT}/features
COPY features/features_std.npy ${LAMBDA_TASK_ROOT}/features
COPY features/features_std_freq.npy ${LAMBDA_TASK_ROOT}/features

COPY models ${LAMBDA_TASK_ROOT}/models
COPY models/model_linear.sav ${LAMBDA_TASK_ROOT}/models
COPY models/model_sigmoid.sav ${LAMBDA_TASK_ROOT}/models
# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]