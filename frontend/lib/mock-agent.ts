// Mock data for agent responses
const codeExamples = {
  python: `def fibonacci(n):
    """Generate fibonacci sequence up to n"""
    a, b = 0, 1
    result = []
    while a < n:
        result.append(a)
        a, b = b, a + b
    return result

print(fibonacci(100))`,
  javascript: `function fetchUserData(userId) {
  return fetch(\`/api/users/\${userId}\`)
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      console.log('User data:', data);
      return data;
    })
    .catch(error => {
      console.error('Error fetching user data:', error);
    });
}`,
  java: `public class UserRepository {
    private final DatabaseConnection connection;
    
    public UserRepository(DatabaseConnection connection) {
        this.connection = connection;
    }
    
    public User findById(Long id) {
        String sql = "SELECT * FROM users WHERE id = ?";
        try (PreparedStatement stmt = connection.prepareStatement(sql)) {
            stmt.setLong(1, id);
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                return mapToUser(rs);
            }
            return null;
        } catch (SQLException e) {
            throw new RepositoryException("Error finding user by id", e);
        }
    }
}`,
}

const documentExamples = [
  {
    title: "Employee Handbook 2024",
    type: "HR Policy",
    date: "January 15, 2024",
    summary: "This document outlines the company's policies, benefits, and expectations for all employees.",
    keyPoints: [
      "Updated remote work policy allows for 3 days of remote work per week",
      "New mental health benefits added to healthcare package",
      "Annual performance reviews will now be conducted quarterly",
      "Updated code of conduct with emphasis on inclusive workplace practices",
    ],
    source: "HR Department / company-policies/employee-handbook-2024.pdf",
  },
  {
    title: "Product Roadmap Q2 2024",
    type: "Product Strategy",
    date: "March 28, 2024",
    summary: "Strategic plan for product development in Q2 2024, including key features and release timelines.",
    keyPoints: [
      "AI-powered recommendation engine scheduled for May release",
      "Mobile app redesign to be completed by end of June",
      "API v2 deprecation planned for mid-Q2",
      "New enterprise tier features to be prioritized",
    ],
    source: "Product Team / roadmaps/q2-2024-roadmap.pptx",
  },
]

const businessExamples = [
  {
    title: "Monthly Active Users (MAU)",
    subtitle: "Last 6 months trend analysis",
    chartType: "line",
    chartData: [
      { name: "Jan", value: 4200 },
      { name: "Feb", value: 4500 },
      { name: "Mar", value: 5100 },
      { name: "Apr", value: 5400 },
      { name: "May", value: 6200 },
      { name: "Jun", value: 7100 },
    ],
    series: [{ name: "Monthly Active Users", dataKey: "value", color: "#3b82f6" }],
    insights: [
      "69% increase in MAU over the last 6 months",
      "Highest growth rate observed after the March product update",
      "Mobile users account for 64% of total active users",
      "Retention rate improved from 72% to 78% in this period",
    ],
  },
  {
    title: "Revenue by Product Category",
    subtitle: "Current quarter breakdown",
    chartType: "pie",
    chartData: [
      { name: "Enterprise", value: 45 },
      { name: "Professional", value: 30 },
      { name: "Basic", value: 15 },
      { name: "Add-ons", value: 10 },
    ],
    insights: [
      "Enterprise tier generates 45% of total revenue",
      "Add-on services show 25% growth compared to previous quarter",
      "Professional tier conversion rate increased by 12%",
      "Average revenue per user (ARPU) is $42 for Basic, $120 for Professional, and $550 for Enterprise",
    ],
  },
  {
    title: "Customer Acquisition Channels",
    subtitle: "Performance comparison",
    chartType: "bar",
    chartData: [
      { name: "Organic Search", acquisition: 320, cost: 0 },
      { name: "Paid Search", acquisition: 280, cost: 14000 },
      { name: "Social Media", acquisition: 240, cost: 9600 },
      { name: "Email", acquisition: 180, cost: 3600 },
      { name: "Referral", acquisition: 120, cost: 2400 },
    ],
    series: [{ name: "New Customers", dataKey: "acquisition", color: "#3b82f6" }],
    insights: [
      "Organic search remains the most cost-effective channel",
      "Social media campaigns show 18% higher conversion rate than last quarter",
      "Email marketing has the lowest customer acquisition cost (CAC) at $20",
      "Referral program shows strong ROI despite lower absolute numbers",
    ],
  },
]

// Function to determine agent type based on user input
function determineAgentType(input, selectedAgent) {
  if (selectedAgent !== "auto") {
    return selectedAgent
  }

  input = input.toLowerCase()

  // Exact matches for demo questions
  if (
    input.includes("which customer acquisition channels are performing best") ||
    input.includes("customer acquisition channels")
  ) {
    return "business"
  }

  if (
    input.includes("show me the monthly active user trends") ||
    input.includes("monthly active user") ||
    input.includes("user trends")
  ) {
    return "business"
  }

  if (
    input.includes("summarize the product roadmap for q2") ||
    input.includes("product roadmap") ||
    input.includes("roadmap for q2")
  ) {
    return "document"
  }

  // Code-related keywords
  if (
    input.includes("code") ||
    input.includes("function") ||
    input.includes("programming") ||
    input.includes("github") ||
    input.includes("repository") ||
    input.includes("python") ||
    input.includes("javascript") ||
    input.includes("java") ||
    input.includes("algorithm") ||
    input.includes("implementation")
  ) {
    return "code"
  }

  // Document-related keywords
  if (
    input.includes("document") ||
    input.includes("policy") ||
    input.includes("handbook") ||
    input.includes("report") ||
    input.includes("memo") ||
    input.includes("summary") ||
    input.includes("guideline") ||
    input.includes("roadmap") ||
    input.includes("summarize")
  ) {
    return "document"
  }

  // Business-related keywords
  if (
    input.includes("chart") ||
    input.includes("graph") ||
    input.includes("data") ||
    input.includes("metrics") ||
    input.includes("analytics") ||
    input.includes("revenue") ||
    input.includes("users") ||
    input.includes("customers") ||
    input.includes("sales") ||
    input.includes("trends") ||
    input.includes("performance") ||
    input.includes("acquisition") ||
    input.includes("active user") ||
    input.includes("monthly")
  ) {
    return "business"
  }

  // If no clear match, try to make a best guess based on the query
  if (input.includes("show") && (input.includes("trends") || input.includes("data"))) {
    return "business"
  }

  if (input.includes("what") && (input.includes("policy") || input.includes("document"))) {
    return "document"
  }

  // Default to code if no clear match (though we could consider a more neutral default)
  return "code"
}

// Function to generate a mock response based on user input
export function mockAgentResponse(input, selectedAgent) {
  const agentType = determineAgentType(input, selectedAgent)
  const id = Date.now().toString()
  const timestamp = new Date().toISOString()

  switch (agentType) {
    case "code":
      // Determine which language to use based on input
      let language = "javascript"
      if (input.toLowerCase().includes("python")) {
        language = "python"
      } else if (input.toLowerCase().includes("java")) {
        language = "java"
      }

      return {
        id,
        role: "assistant",
        content: "Here's the code you requested:",
        timestamp,
        agentType: "code",
        agentData: {
          code: codeExamples[language],
          language: language,
          explanation: "This code demonstrates a basic implementation that you can use as a starting point.",
        },
      }

    case "document":
      // Select a document example based on input
      let docIndex = 0
      if (
        input.toLowerCase().includes("roadmap") ||
        input.toLowerCase().includes("product") ||
        input.toLowerCase().includes("q2")
      ) {
        docIndex = 1
      } else {
        docIndex = 0 // Default to employee handbook
      }

      return {
        id,
        role: "assistant",
        content: "I found this document that might help answer your question:",
        timestamp,
        agentType: "document",
        agentData: documentExamples[docIndex],
      }

    case "business":
      // Select a business chart example based on input
      let chartIndex = 0

      if (
        input.toLowerCase().includes("monthly active user") ||
        input.toLowerCase().includes("user trends") ||
        input.toLowerCase().includes("mau")
      ) {
        chartIndex = 0 // Monthly Active Users chart
      } else if (
        input.toLowerCase().includes("revenue") ||
        input.toLowerCase().includes("product category") ||
        input.toLowerCase().includes("breakdown")
      ) {
        chartIndex = 1 // Revenue by Product Category chart
      } else if (
        input.toLowerCase().includes("acquisition") ||
        input.toLowerCase().includes("channel") ||
        input.toLowerCase().includes("performing best")
      ) {
        chartIndex = 2 // Customer Acquisition Channels chart
      }

      return {
        id,
        role: "assistant",
        content: "Here's the business data visualization you requested:",
        timestamp,
        agentType: "business",
        agentData: businessExamples[chartIndex],
      }

    default:
      return {
        id,
        role: "assistant",
        content:
          "I'm not sure how to help with that specific request. Could you try rephrasing or selecting a specific agent type?",
        timestamp,
      }
  }
}
