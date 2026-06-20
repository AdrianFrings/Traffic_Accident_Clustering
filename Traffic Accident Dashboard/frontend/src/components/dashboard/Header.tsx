import { ArrowLeft, Github } from 'lucide-react';
import { Button } from '../ui/button';
import { ThemeToggle } from '../theme-toggle';

const PORTFOLIO_URL = 'https://adrianfrings.com';
const REPO_URL = 'https://github.com/AdrianFrings/Traffic_Accident_Clustering';

export function Header() {
  return (
    <header className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-lg border-b border-border">
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-4 min-w-0">
            <a
              href={PORTFOLIO_URL}
              className="hidden sm:flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ArrowLeft className="mr-1.5 h-4 w-4" />
              Portfolio
            </a>
            <div className="h-5 w-px bg-border hidden sm:block" />
            <div className="min-w-0">
              <div className="text-base sm:text-lg tracking-wide truncate">
                Traffic Accident Risk
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" asChild className="hidden sm:inline-flex">
              <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
                <Github className="h-4 w-4 mr-2" />
                Source
              </a>
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );
}
