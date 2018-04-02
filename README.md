# Chatbot

docker-compose -f docker-compose-chatbot.yml build

docker-compose -f docker-compose-chatbot.yml up

curl -X POST -H "Content-Type:application/json" --data '{"query":"Who is mahatma gandhi?"}' http://127.0.0.1:6006/api/v1/schemas/score
