import { ArrowLeft, Github } from 'lucide-react';

const PORTFOLIO_URL = 'https://adrianfrings.com';
const REPO_URL = 'https://github.com/AdrianFrings/Traffic_Accident_Clustering';

export function Header() {
  return (
    <header className="max-w-7xl mx-auto px-4 pt-5 pb-2 flex items-center justify-between gap-4">
      <a href={PORTFOLIO_URL} className="mono-caps flex items-center hover:text-[var(--ink)] transition-colors">
        <ArrowLeft className="mr-1.5 h-3.5 w-3.5" />
        adrianfrings.com
      </a>
      <a
        href={REPO_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="mono-caps flex items-center border border-[var(--rule)] px-3 py-1.5 hover:bg-[var(--orange)] hover:border-[var(--orange)] hover:text-[#100b07] transition-colors"
      >
        <Github className="h-3.5 w-3.5 mr-2" />
        Source
      </a>
    </header>
  );
}
