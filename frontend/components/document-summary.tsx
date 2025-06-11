export function DocumentSummary({ data }) {
  return (
    <div className="mt-4 border rounded-md overflow-hidden">
      <div className="bg-purple-50 dark:bg-purple-900/20 border-b px-4 py-2">
        <h3 className="font-medium">{data.title}</h3>
        <div className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
          <span>{data.type}</span>
          <span>•</span>
          <span>{data.date}</span>
        </div>
      </div>
      <div className="p-4 space-y-3">
        {data.summary && (
          <div>
            <h4 className="text-sm font-medium mb-1">Summary</h4>
            <p className="text-sm">{data.summary}</p>
          </div>
        )}

        {data.keyPoints && data.keyPoints.length > 0 && (
          <div>
            <h4 className="text-sm font-medium mb-1">Key Points</h4>
            <ul className="list-disc list-inside text-sm space-y-1">
              {data.keyPoints.map((point, index) => (
                <li key={index}>{point}</li>
              ))}
            </ul>
          </div>
        )}

        {data.source && (
          <div className="text-xs text-slate-500 dark:text-slate-400 pt-2 border-t">Source: {data.source}</div>
        )}
      </div>
    </div>
  )
}
