"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";

interface ApiStatusProps {
  apiUrl: string;
}

export function ApiStatus({ apiUrl }: ApiStatusProps) {
  const [status, setStatus] = useState<"loading" | "online" | "offline">("loading");
  const [pingInterval, setPingInterval] = useState<NodeJS.Timeout | null>(null);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${apiUrl}/`);
      if (response.ok) {
        setStatus("online");
      } else {
        setStatus("offline");
      }
    } catch (error) {
      setStatus("offline");
    }
  };

  useEffect(() => {
    checkApiStatus();
    
    // Set up interval to check API status every 30 seconds
    const interval = setInterval(() => {
      checkApiStatus();
    }, 30000);
    
    setPingInterval(interval);
    
    return () => {
      if (pingInterval) clearInterval(pingInterval);
    };
  }, [apiUrl]);

  return (
    <div className="flex items-center gap-2 text-sm">
      <div
        className={`w-2 h-2 rounded-full ${
          status === "online"
            ? "bg-green-400"
            : status === "offline"
            ? "bg-red-400"
            : "bg-yellow-400"
        }`}
      />
      <span className="text-purple-100">
        API Status:{" "}
        {status === "online"
          ? "Online"
          : status === "offline"
          ? "Offline"
          : "Checking..."}
      </span>
      {status === "offline" && (
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-2 ml-2 border-purple-400 text-purple-200 hover:bg-purple-800"
          onClick={checkApiStatus}
        >
          Retry
        </Button>
      )}
    </div>
  );
}
