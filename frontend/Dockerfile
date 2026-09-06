# Stage 1: build the static bundle
FROM node:22-slim AS builder

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ ./
# VITE_API_BASE_URL is baked in at build time (Vite inlines env vars into
# the bundle). Pass it via --build-arg in CI/CD to point at the real API.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:1.27-alpine AS runtime

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
