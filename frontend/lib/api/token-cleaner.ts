/**
 * Token cleaner utility for filtering out routing directives from the LangGraph agents
 * This prevents internal routing instructions from showing up in the UI
 */

/**
 * Clean a streaming token to remove routing directives
 * @param token The token from the streaming API
 * @returns Cleaned token without routing directives
 */
export function cleanStreamingToken(token: string): string {
  // Don't process empty tokens
  if (!token || token.trim().length === 0) {
    return token;
  }
  
  // Routing directive patterns that should be filtered out
  const routingPatterns = [
    // Agent transfer directives with variations in formatting and spacing
    /Transfer\s+to\s+\w+\s+AGENT/i,
    /Transfer\s+to\s+DOCUMENT\s+RAG\s+AGENT/i,
    /Transfer\s+to\s+DATA\s+ANALYTICS\s+AGENT/i,
    /Transfer\s+to\s+CODE\/CONVERSATION\s+AGENT/i,
    
    // Shorter "to AGENT" variations (including with quotation marks)
    /"?\s*to\s+\w+\s+AGENT\s*"?/i,
    /"?\s*to\s+RAG\s+AGENT\s*"?/i,
    /"?\s*to\s+DOCUMENT\s+RAG\s+AGENT\s*"?/i,
    /"?\s*to\s+DATA\s+ANALYTICS\s+AGENT\s*"?/i,
    /"?\s*to\s+CODE\/CONVERSATION\s+AGENT\s*"?/i,
    
    // Just agent names with optional quotes
    /"?\s*RAG\s+AGENT\s*"?/i,
    /"?\s*DOCUMENT\s+RAG\s*"?/i,
    
    // Bold formatting variations
    /\*\*Transfer\s+to\s+\w+\s+AGENT\*\*/i,
    /\*\*to\s+\w+\s+AGENT\*\*/i,
    
    // Agent names as standalone phrases at beginning of lines
    /^"?\s*DOCUMENT\s+RAG\s+AGENT\s*"?\s*$/i,
    /^"?\s*DATA\s+ANALYTICS\s+AGENT\s*"?\s*$/i,
    /^"?\s*CODE\/CONVERSATION\s+AGENT\s*"?\s*$/i,
    /^"?\s*RAG\s+AGENT\s*"?\s*$/i,
    /^"?\s*SUPERVISOR\s*"?\s*$/i,
    
    // Finish directives
    /"?\s*\bFINISH\b\s*"?/i,
    /\*\*FINISH\*\*/i,
    
    
    // Korean variations
    /전달.*에이전트/i,
    /전환.*에이전트/i,
    
    // Common routing phrases
    /^"?\s*Routing\s+to/i,
    /^"?\s*Transferring\s+to/i,
    /^"?\s*Handing\s+off\s+to/i
  ];
  
  // Check if token contains any routing patterns
  for (const pattern of routingPatterns) {
    if (pattern.test(token)) {
      return ''; // Filter out tokens containing routing directives
    }
  }
  
  // Check for suspicious tokens that might be part of a routing directive
  const suspiciousTokens = ['transfer', 'agent', 'rag', 'analytics', 'finish', 'document', 'routing'];
  if (suspiciousTokens.some(suspect => token.toLowerCase().includes(suspect))) {
    // For suspicious tokens, we check if they are likely part of a routing command
    if (token.length < 15) {
      return ''; // Filter out short suspicious tokens as they're likely part of a directive
    }
  }

  return token;
}

/**
 * Cleans a full message content to remove routing directives
 * @param content The full message content
 * @returns Cleaned message content
 */
export function cleanFullMessageContent(content: string): string {
  if (!content) return content;
  
  // Remove exact routing directive phrases
  let cleaned = content;
  
  const directivePatterns = [
    // Transfer directives with quotation handling
    /Transfer\s+to\s+\w+\s+AGENT/gi,
    /Transfer\s+to\s+DOCUMENT\s+RAG\s+AGENT/gi,
    /Transfer\s+to\s+DATA\s+ANALYTICS\s+AGENT/gi,
    /Transfer\s+to\s+CODE\/CONVERSATION\s+AGENT/gi,
    /\*\*Transfer\s+to\s+\w+\s+AGENT\*\*/gi,
    
    // Shorter variants with quotation handling
    /to\s+\w+\s+AGENT/gi,
    /to\s+RAG\s+AGENT/gi,
    /to\s+DOCUMENT\s+RAG\s+AGENT/gi,
    /to\s+DATA\s+ANALYTICS\s+AGENT/gi,
    /to\s+CODE\/CONVERSATION\s+AGENT/gi,
    
    // Finish directive
    /\bFINISH\b/gi,
    /\*\*FINISH\*\*/gi,
    
    // Korean variants
    /전달.*에이전트/gi,
    /전환.*에이전트/gi
  ];
  
  // Replace all directive patterns with empty string
  for (const pattern of directivePatterns) {
    cleaned = cleaned.replace(pattern, '');
  }
  
  // Handle quotation marks and other artifacts after removing directives
  cleaned = cleaned.replace(/"\s*"/g, ''); // Remove empty quotes with whitespace between
  cleaned = cleaned.replace(/"\s*$/g, ''); // Remove trailing quotation mark
  cleaned = cleaned.replace(/^\s*"/g, ''); // Remove leading quotation mark
  cleaned = cleaned.replace(/\n"\s*\n/g, '\n'); // Remove quotes on their own line
  cleaned = cleaned.replace(/\n+"\s*$/g, ''); // Remove quotes at the end after newlines
  
  // Remove short lines that are likely just routing directives
  const lines = cleaned.split('\n');
  const filteredLines = lines.filter(line => {
    const lineLower = line.toLowerCase();
    
    // Check patterns for common routing or agent-related lines
    const linePatterns = [
      /document\s+rag\s+agent/i,
      /analytics\s+agent/i,
      /code\s+agent/i,
      /transfer.*agent/i,
      /\*\*agent\*\*/i,
      /^routing\s+to/i,
      /^transferring\s+to/i,
      /^final\s+decision/i
    ];
    
    const isRoutingLine = linePatterns.some(pattern => pattern.test(lineLower));
    
    // If it looks like a routing line and it's short, filter it out
    if (isRoutingLine && line.trim().length < 80) {
      return false;
    }
    
    return true;
  });
  
  // Rejoin and clean up formatting
  cleaned = filteredLines.join('\n');
  
  // Remove multiple consecutive blank lines
  cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
  
  // Remove trailing "" if present
  cleaned = cleaned.replace(/\n*""\s*$/g, '');
  
  // Remove any trailing empty quotation pairs
  cleaned = cleaned.replace(/\n*["']{1,2}\s*$/g, '');
  
  // Remove punctuation or whitespace at the beginning
  cleaned = cleaned.replace(/^[\s,.;:\-]+/, '');
  
  // Final cleanup pass - if message ends with just a quote, remove it
  cleaned = cleaned.replace(/\s*"\s*$/g, '');
  
  // Clean up any potential empty lines at the end
  cleaned = cleaned.replace(/\n+\s*$/g, '');
  
  return cleaned.trim();
}
