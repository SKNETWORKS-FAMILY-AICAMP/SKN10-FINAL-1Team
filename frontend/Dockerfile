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

# Build the Next.js application (optional for dev mode)
# RUN npm run build

# Remove devDependencies after build (skip for dev mode)
# RUN npm prune --production

# Expose the port the app runs on
EXPOSE 3000

# Run the application in development mode for django-nextjs integration
CMD ["npm", "run", "dev"] 