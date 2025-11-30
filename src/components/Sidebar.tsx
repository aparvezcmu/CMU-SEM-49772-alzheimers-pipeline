import React from 'react';
import { FileText, Folder, Settings, HelpCircle, Brain } from 'lucide-react';

export type SidebarSection = 'notebooks' | 'files' | 'settings' | null;

interface SidebarProps {
  activeSection: SidebarSection;
  onSectionChange: (section: SidebarSection) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeSection, onSectionChange }) => {
  const sections = [
    { id: 'notebooks' as const, icon: FileText, label: 'Notebooks', tooltip: 'Jupyter Notebooks' },
    { id: 'files' as const, icon: Folder, label: 'Files', tooltip: 'Workspace Files' },
    { id: 'settings' as const, icon: Settings, label: 'Settings', tooltip: 'Settings' },
  ];

  return (
    <div className="w-12 bg-gray-800 flex flex-col items-center py-4 gap-2">
      {/* Logo/Title */}
      <div className="mb-4 pb-4 border-b border-gray-700 w-full flex justify-center" title="Alzheimer's Pipeline">
        <Brain className="w-7 h-7 text-purple-400" />
      </div>

      {/* Section Icons */}
      {sections.map(({ id, icon: Icon, label, tooltip }) => (
        <button
          key={id}
          onClick={() => onSectionChange(activeSection === id ? null : id)}
          className={`
            relative group w-10 h-10 rounded-lg flex items-center justify-center
            transition-colors duration-200
            ${activeSection === id
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:bg-gray-700 hover:text-white'
            }
          `}
          aria-label={label}
          title={tooltip}
        >
          <Icon className="w-5 h-5" />

          {/* Tooltip */}
          <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50 transition-opacity">
            {tooltip}
          </div>
        </button>
      ))}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Help Icon */}
      <button
        className="w-10 h-10 rounded-lg flex items-center justify-center text-gray-400 hover:bg-gray-700 hover:text-white transition-colors group relative"
        title="Help"
        aria-label="Help"
      >
        <HelpCircle className="w-5 h-5" />

        {/* Tooltip */}
        <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50 transition-opacity">
          Help
        </div>
      </button>
    </div>
  );
};
