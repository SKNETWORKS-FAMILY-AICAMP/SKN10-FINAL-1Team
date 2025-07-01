# Use an official Node.js runtime as a parent image
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies (including devDependencies for build)
RUN npm ci

# Copy the rest of the application code
COPY . .
# 빌드 단계에 환경 변수 추가
ENV NEXT_PUBLIC_BACKEND_URL=http://sknhinton.us-east-1.elasticbeanstalk.com

# Build the Next.js application for production
RUN npm run build

# Remove devDependencies after build for smaller image size
RUN npm prune --production

# Expose the port the app runs on
EXPOSE 3000

# Run the application in production mode
CMD ["npm", "start"]