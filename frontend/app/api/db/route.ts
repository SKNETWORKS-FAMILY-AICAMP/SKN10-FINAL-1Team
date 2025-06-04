import { NextResponse } from 'next/server';
import { Pool } from 'pg';
import { type NextRequest } from 'next/server';

// Create a connection pool
const pool = new Pool({
  connectionString: process.env.POSTGRES_CONNECTION_STRING,
});

// Define the route handler for POST requests
export async function POST(request: NextRequest) {
  try {
    // Verify user authentication using JWT token
    const authorization = request.headers.get('authorization') || '';
    
    if (!authorization || !authorization.startsWith('Bearer ')) {
      console.error('Unauthorized database access attempt: No valid token');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    
    // In a production environment, we would verify the JWT token here
    // For now, we'll just check if it exists
    
    const body = await request.json();
    const { sql, params } = body;
    
    // Security checks
    if (!sql) {
      return NextResponse.json({ error: 'No SQL provided' }, { status: 400 });
    }
    
    // Debug log the SQL query
    console.log('Processing SQL query:', sql);
    console.log('With params:', params);
    
    // Validate SQL to prevent injection attacks
    // Remove DELETE from forbidden patterns since we need it for chat operations
    const forbiddenPatterns = [
      /DROP\s+/i,
      /ALTER\s+/i,
      /TRUNCATE\s+/i,
      /GRANT\s+/i,
      /REVOKE\s+/i,
    ];
    
    // Only allow specific database operations for chat data
    const allowedOperations = [
      // Check if SQL is for chat sessions or messages tables
      /SELECT.*FROM[\s\n]+chat_sessions/i,
      /SELECT.*FROM[\s\n]+chat_messages/i,
      /INSERT.*INTO[\s\n]+chat_sessions/i,
      /INSERT.*INTO[\s\n]+chat_messages/i,
      /UPDATE[\s\n]+chat_sessions/i,
      /DELETE.*FROM[\s\n]+chat_sessions/i,
      /DELETE.*FROM[\s\n]+chat_messages/i,
      /ORDER[\s\n]+BY/i
    ];
    
    // Debug log the SQL validation checks
    console.log('Checking forbidden patterns...');
    
    // Check for forbidden patterns
    const hasForbiddenPattern = forbiddenPatterns.some(pattern => {
      const matches = pattern.test(sql);
      if (matches) {
        console.error(`Forbidden pattern matched: ${pattern}`);
      }
      return matches;
    });
    
    if (hasForbiddenPattern) {
      console.error('Forbidden SQL operation attempted:', sql);
      return NextResponse.json({ error: 'Operation not allowed' }, { status: 403 });
    }
    
    // Debug log the SQL allowlist check
    console.log('Checking allowed operations...');
    
    // Ensure it's a valid allowed operation
    const isAllowedOperation = allowedOperations.some(pattern => {
      const matches = pattern.test(sql);
      if (matches) {
        console.log(`Allowed pattern matched: ${pattern}`);
      }
      return matches;
    });
    
    if (!isAllowedOperation) {
      console.error('SQL operation not in allowlist:', sql);
      return NextResponse.json({ error: 'Operation not allowed' }, { status: 403 });
    }
    
    // Execute the query
    const client = await pool.connect();
    try {
      const result = await client.query(sql, params);
      return NextResponse.json({ rows: result.rows });
    } finally {
      client.release();
    }
  } catch (error) {
    console.error('Database error:', error);
    return NextResponse.json({ error: 'Database operation failed' }, { status: 500 });
  }
}
