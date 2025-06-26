import { useEffect, useRef } from "react";

interface ChartSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  chartContent: any;
}

export function ChartSidebar({ isOpen, onClose, chartContent }: ChartSidebarProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && chartContent && chartRef.current) {
      chartRef.current.innerHTML = chartContent.canvas_html || "";
      try {
        // Chart.js가 없으면 CDN으로 동적 로드
        if (!(window as any).Chart) {
          const script = document.createElement("script");
          script.src = "https://cdn.jsdelivr.net/npm/chart.js";
          script.onload = () => {
            // eslint-disable-next-line no-new-func
            new Function(chartContent.script_js)();
          };
          document.body.appendChild(script);
        } else {
          // eslint-disable-next-line no-new-func
          new Function(chartContent.script_js)();
        }
      } catch (e) {
        if (chartRef.current) {
          chartRef.current.innerHTML = "<div class='text-red-500'>차트 스크립트 실행 오류</div>";
        }
      }
    }
  }, [isOpen, chartContent]);

  return (
    <div
      className={`fixed top-0 right-0 h-full w-full max-w-md bg-white shadow-2xl z-50 border-l border-gray-200 flex flex-col transition-transform duration-300 ${
        isOpen ? "translate-x-0" : "translate-x-full"
      }`}
      style={{ minWidth: 350 }}
    >
      <div className="flex justify-between items-center p-4 border-b border-gray-200">
        <h3 className="text-lg font-bold text-gray-800">Chart View</h3>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
          닫기
        </button>
      </div>
      <div ref={chartRef} className="flex-1 overflow-y-auto p-4" />
    </div>
  );
} 