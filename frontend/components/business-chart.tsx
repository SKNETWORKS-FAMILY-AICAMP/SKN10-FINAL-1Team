"use client"

import { Card } from "@/components/ui/card"
import { ChartTooltip } from "@/components/ui/chart"
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts"

export function BusinessChart({ data }) {
  // Determine chart type based on data
  const renderChart = () => {
    switch (data.chartType) {
      case "bar":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data.chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <ChartTooltip />
              {data.series.map((series, index) => (
                <Bar
                  key={index}
                  dataKey={series.dataKey}
                  fill={series.color || `hsl(${index * 40}, 70%, 50%)`}
                  name={series.name}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )

      case "line":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data.chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <ChartTooltip />
              {data.series.map((series, index) => (
                <Line
                  key={index}
                  type="monotone"
                  dataKey={series.dataKey}
                  stroke={series.color || `hsl(${index * 40}, 70%, 50%)`}
                  name={series.name}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )

      case "pie":
        const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"]
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data.chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                nameKey="name"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {data.chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <ChartTooltip />
            </PieChart>
          </ResponsiveContainer>
        )

      default:
        return <div className="p-4 text-center text-slate-500">Chart visualization not available</div>
    }
  }

  return (
    <Card className="mt-4 overflow-hidden">
      <div className="bg-green-50 dark:bg-green-900/20 border-b px-4 py-2">
        <h3 className="font-medium">{data.title}</h3>
        {data.subtitle && <p className="text-xs text-slate-500 dark:text-slate-400">{data.subtitle}</p>}
      </div>

      <div className="p-4">
        {renderChart()}

        {data.insights && data.insights.length > 0 && (
          <div className="mt-4 pt-4 border-t">
            <h4 className="text-sm font-medium mb-2">Key Insights</h4>
            <ul className="list-disc list-inside text-sm space-y-1">
              {data.insights.map((insight, index) => (
                <li key={index}>{insight}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  )
}
