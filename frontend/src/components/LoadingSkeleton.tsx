import React from 'react';

interface SkeletonProps {
  className?: string;
  height?: string;
  width?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({ 
  className = '', 
  height = 'h-4', 
  width = 'w-full' 
}) => (
  <div className={`animate-pulse bg-gray-700 rounded ${height} ${width} ${className}`}></div>
);

export const SkeletonCard: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
    <div className="space-y-4">
      <Skeleton height="h-4" width="w-3/4" />
      <Skeleton height="h-8" width="w-1/2" />
      <div className="flex space-x-4">
        <Skeleton height="h-6" width="w-20" />
        <Skeleton height="h-6" width="w-20" />
      </div>
    </div>
  </div>
);

export const SkeletonTable: React.FC<{ rows?: number; columns?: number }> = ({ 
  rows = 5, 
  columns = 4 
}) => (
  <div className="space-y-3">
    {Array.from({ length: rows }).map((_, i) => (
      <div key={i} className="flex space-x-4">
        {Array.from({ length: columns }).map((_, j) => (
          <Skeleton key={j} height="h-4" width="w-full" />
        ))}
      </div>
    ))}
  </div>
);

export const SkeletonStats: React.FC = () => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    {Array.from({ length: 4 }).map((_, i) => (
      <SkeletonCard key={i} />
    ))}
  </div>
);

export const SkeletonChart: React.FC = () => (
  <div className="bg-gray-800 rounded-lg p-6">
    <Skeleton height="h-6" width="w-1/3" className="mb-4" />
    <div className="space-y-2">
      {Array.from({ length: 8 }).map((_, i) => (
        <Skeleton 
          key={i} 
          height="h-3" 
          width={`w-${Math.floor(Math.random() * 40) + 20}%`} 
        />
      ))}
    </div>
  </div>
);

export const SkeletonList: React.FC<{ items?: number }> = ({ items = 3 }) => (
  <div className="space-y-4">
    {Array.from({ length: items }).map((_, i) => (
      <div key={i} className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center space-x-4">
          <Skeleton height="h-10" width="w-10" className="rounded-full" />
          <div className="flex-1 space-y-2">
            <Skeleton height="h-4" width="w-1/2" />
            <Skeleton height="h-3" width="w-1/3" />
          </div>
          <Skeleton height="h-6" width="w-16" />
        </div>
      </div>
    ))}
  </div>
);
