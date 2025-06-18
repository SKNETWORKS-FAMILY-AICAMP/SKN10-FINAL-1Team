"use client"

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { GithubIcon } from 'lucide-react';

interface GithubAuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onTokenSubmit: (token: string) => Promise<void>;
  onOAuthStart: () => void; // OAuth 로직은 추후 구현
}

export default function GithubAuthModal({
  isOpen,
  onClose,
  onTokenSubmit,
  onOAuthStart,
}: GithubAuthModalProps) {
  const [token, setToken] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmitToken = async () => {
    if (!token.trim()) {
      setError('GitHub 토큰을 입력해주세요.');
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      await onTokenSubmit(token);
      // 성공 시 모달이 닫히거나 다른 UI 변경은 부모 컴포넌트에서 처리
    } catch (err) {
      setError('토큰 인증에 실패했습니다. 다시 시도해주세요.');
      console.error('Token submission failed:', err);
    }
    setIsLoading(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="flex items-center">
            <GithubIcon className="mr-2 h-5 w-5" /> GitHub 계정 연동
          </DialogTitle>
          <DialogDescription>
            GitHub Personal Access Token을 입력하거나 GitHub 계정으로 직접 로그인하여 연동할 수 있습니다.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="github-token">Personal Access Token</Label>
            <Input
              id="github-token"
              type="password"
              placeholder="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              disabled={isLoading}
            />
            {error && <p className="text-sm text-red-500">{error}</p>}
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              토큰은 <a href="https://github.com/settings/tokens" target="_blank" rel="noopener noreferrer" className="underline">GitHub 설정</a>에서 생성할 수 있으며, 'repo' 권한이 필요합니다.
            </p>
          </div>
          <Button onClick={handleSubmitToken} disabled={isLoading || !token.trim()}>
            {isLoading ? '연동 중...' : '토큰으로 연결'}
          </Button>
          <div className="relative my-4">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-background px-2 text-muted-foreground">
                또는
              </span>
            </div>
          </div>
          <Button variant="outline" onClick={onOAuthStart} disabled={isLoading}>
            <GithubIcon className="mr-2 h-4 w-4" /> GitHub으로 로그인
          </Button>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose} disabled={isLoading}>
            취소
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
