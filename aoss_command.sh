#!bin/sh
source .env

# Check the index names
curl  --aws-sigv4 "aws:amz:eu-west-1:es" \
  --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}"\
  -H "x-amz-security-token: ${AWS_SESSION_TOKEN}"\
  -XGET "${AOSS_ENDPOINT}/_aliases?pretty=true"

# Check the index details
curl  --aws-sigv4 "aws:amz:eu-west-1:es" \
  --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}"\
  -H "x-amz-security-token: ${AWS_SESSION_TOKEN}"\
  -XGET "${AOSS_ENDPOINT}/aoss_qa?pretty=true"

# Count number of documents
curl  --aws-sigv4 "aws:amz:eu-west-1:es" \
  --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}"\
  -H "x-amz-security-token: ${AWS_SESSION_TOKEN}"\
  -XGET "${AOSS_ENDPOINT}/aoss_qa/_count?"

# Delete the indexes
curl  --aws-sigv4 "aws:amz:eu-west-1:es" \
  --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}"\
  -H "x-amz-security-token: ${AWS_SESSION_TOKEN}"\
  -XDELETE "${AOSS_ENDPOINT}/aoss_qa"